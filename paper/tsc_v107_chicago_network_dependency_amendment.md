# TSC v107: Chicago network-build dependency amendment

Date frozen: 2026-08-31 (Asia/Shanghai), before the v2 static scan.

## v1 outcome

The first network-build implementation successfully ran SUMO 1.22
`netconvert` on the frozen Chicago OSM source and wrote
`chicago_full.net.xml.gz`, but the subsequent anchor scan stopped before its
first coordinate conversion because `sumolib.convertXY2LonLat` imported an
unavailable `pyproj` module. The preserved failure is
`network_v1.rejected/build_failure.json` with
`ModuleNotFoundError: No module named 'pyproj'`. No anchor inventory, safety
simulation, source-selection result, controller rollout, or efficacy result was
available.

This is an execution-dependency failure, not a failed Chicago topology or
safety gate. The v1 network file was produced by the frozen command, its
`netconvert.stdout.log` is exactly `Success.`, its size is `362471296` bytes,
and its SHA-256 is
`45250772dcff2cd79bd4472e0657d09c20eb32be280a1099d5ba93b831ac42ae`.

## Frozen v2 correction

The v2 static scan makes two implementation-only corrections:

1. activate the already frozen cluster dependency `pyproj==3.7.1` from the
   official CPython 3.10 manylinux wheel with SHA-256
   `1e47c4e93b88d99dd118875ee3ca0171932444cdc0b52d493371b5d98d0f30ee`;
2. replace the memory-heavy `sumolib.readNet` object expansion with one gzip XML
   pass that extracts passenger-permitted edges, lane attributes, connections,
   locations, junctions, and traffic-light programs, followed by one vectorized
   inverse projection and the same deterministic SCC and farthest-anchor rules.

The v2 build adopts the byte-identical successful v1 network only after checking
the preserved failure identity, success log, size, and SHA-256. It does not rerun
or change OSM conversion. Passenger permission follows SUMO lane `allow` and
`disallow` attributes; default lanes permit passenger vehicles. Connections are
the published SUMO `from`/`to` edge transitions. Edge midpoint, minimum length,
capacity score, largest strongly connected component, 32 anchors per community
area, and all 77-area requirements remain exactly as frozen in the parent
protocol.

No source, date, row, edge, signal, netconvert option, routing rule, demand rule,
seed, safety threshold, or efficacy threshold changes under this amendment. A
v2 static failure still rejects the candidate before demand expansion.
