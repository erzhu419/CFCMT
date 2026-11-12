"""Static lane/link topology for the declared pure-Python packet model.

Connection dictionaries retain XML order and physical paths. TLS link indices
are local joins: several physical movements may share one signal link, and an
internal lane may belong to several complete movements. Neither case selects
an individual vehicle's route.
"""
from collections import defaultdict


class RoadTopology:
    def __init__(self, static):
        self.lanes = static['lanes']
        self.successors = static['successors']
        self.predecessors = static['predecessors']
        self._roads = defaultdict(list)
        for lane, geometry in self.lanes.items():
            if not geometry['internal']:
                self._roads[geometry['edge_id']].append(lane)
        self._roads = {edge: tuple(sorted(lanes, key=lambda lane: self.lanes[lane]['index']))
                       for edge, lanes in self._roads.items()}
        self._lane_indices = {edge: {self.lanes[lane]['index']: lane for lane in lanes}
                              for edge, lanes in self._roads.items()}
        connections, origins, internals = [], defaultdict(list), defaultdict(list)
        self.by_key, self._offsets = {}, {}
        self.pp_lane_by_link = defaultdict(dict)
        for index, row in enumerate(static['connections']):
            origin, destination = row['from_lane'], row['to_lane']
            if self.lanes[origin]['internal']:
                continue  # Continuation rows form paths; they are not extra actions.
            if self.lanes[destination]['internal']:
                raise ValueError('A whole movement must end at an ordinary receiving lane')
            paths = self._complete_paths(origin, row['via'] or destination, destination)
            if len(paths) != 1:
                raise ValueError(f'Ordinary movement has {len(paths)} complete internal paths: {origin} -> {destination}')
            connection = dict(key=f'conn_{index}', from_lane=origin, to_lane=destination,
                path=paths[0], tls_id=row['tls_id'], link_index=row['link_index'],
                direction=row['direction'], from_edge=self.lanes[origin]['edge_id'],
                to_edge=self.lanes[destination]['edge_id'])
            self.by_key[connection['key']] = connection
            connections.append(connection)
            origins[origin].append(connection)
            offsets, distance = {}, 0.
            for lane in connection['path']:
                offsets[lane] = distance
                distance += self.lanes[lane]['length_m']
            self._offsets[connection['key']] = offsets
            for lane in connection['path'][1:-1]:
                internals[lane].append(connection)
            tls, link = connection['tls_id'], connection['link_index']
            if tls is not None:
                self.pp_lane_by_link[tls].setdefault(link, origin)
        self.connections = tuple(connections)
        self._from = {lane: tuple(rows) for lane, rows in origins.items()}
        self.internal_connections = {lane: tuple(internals[lane]) for lane, row in self.lanes.items()
                                     if row['internal']}
        self.internal_progress = {lane: {c['key']: self._offsets[c['key']][lane] for c in owners}
                                  for lane, owners in self.internal_connections.items()}
        self.pp_lane_by_link = {tls: dict(rows) for tls, rows in self.pp_lane_by_link.items()}
        # An explicit getControlledLinks export wins over inferred XML order.
        if 'pp_lane_by_link' in static:
            self.pp_lane_by_link = {tls: {int(link): lane for link, lane in rows.items()}
                                    for tls, rows in static['pp_lane_by_link'].items()}
        self._foes, self._yields = {}, {}
        for tls in self.pp_lane_by_link:
            context = static['conflicts_by_tls'][tls]
            if context.get('full_context_complete') is False:
                raise ValueError(f'Incomplete controlled conflict context: {tls}')
            self._foes[tls] = {tuple(pair) for pair in context['foes']}
            self._yields[tls] = {tuple(pair) for pair in context['response']}

    def _complete_paths(self, origin, first, destination):
        paths = []

        def visit(path):
            current = path[-1]
            if current == destination:
                paths.append(tuple(path))
            elif self.lanes[current]['internal']:
                for following in self.successors[current]:
                    if following not in path:
                        visit(path+[following])

        visit([origin, first])
        return paths

    def _connection(self, connection):
        return self.by_key[connection] if isinstance(connection, str) else connection

    def connections_from(self, lane):
        """Whole movements originating at an ordinary lane, in XML order."""
        return self._from.get(lane, ())

    def compatible(self, lane, target_edge):
        """Return all physical connections matching the persistent target road."""
        return tuple(c for c in self.connections_from(lane) if c['to_edge'] == target_edge)

    def road_lanes(self, edge):
        """Ordinary-road lanes ordered by SUMO lane index."""
        return self._roads.get(edge, ())

    def adjacent(self, lane, delta):
        """Index +1 is left, -1 right, 0 stay; internal lanes have no lateral action."""
        if delta not in (-1, 0, 1):
            raise ValueError('A lateral action is left, stay or right')
        geometry = self.lanes[lane]
        if geometry['internal']:
            return None
        return self._lane_indices[geometry['edge_id']].get(geometry['index']+delta)

    def path_progress(self, connection, lane, position_m):
        """Front distance from the start of this movement's ordinary origin lane."""
        connection = self._connection(connection)
        return self._offsets[connection['key']][lane]+float(position_m)

    def path_distance(self, connection, from_lane, from_position_m, to_lane, to_position_m):
        """Signed downstream distance along one explicitly selected movement."""
        return (self.path_progress(connection, to_lane, to_position_m)
                - self.path_progress(connection, from_lane, from_position_m))

    def foes(self, first, second):
        first, second = self._connection(first), self._connection(second)
        tls = first['tls_id']
        return (tls is not None and tls == second['tls_id']
                and (first['link_index'], second['link_index']) in self._foes[tls])

    def yields_to(self, first, second):
        """response[i,j] means connection i yields to connection j."""
        first, second = self._connection(first), self._connection(second)
        tls = first['tls_id']
        return (tls is not None and tls == second['tls_id']
                and (first['link_index'], second['link_index']) in self._yields[tls])

    def metadata(self):
        """Small coverage/ambiguity counts; never choose an internal lane's route."""
        owners = self.internal_connections
        return dict(lanes=len(self.lanes), ordinary_roads=len(self._roads),
            whole_movements=len(self.connections), internal_lanes=len(owners),
            covered_internal_lanes=sum(bool(rows) for rows in owners.values()),
            shared_internal_lanes=sum(len(rows) > 1 for rows in owners.values()),
            shared_internal_different_receivers=sum(len({c['to_lane'] for c in rows}) > 1
                                                   for rows in owners.values()),
            multiple_movements_per_link=sum(sum(c['tls_id'] == tls and c['link_index'] == link
                for c in self.connections) > 1 for tls, rows in self.pp_lane_by_link.items() for link in rows))
