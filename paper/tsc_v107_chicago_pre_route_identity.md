# TSC v107: Chicago pre-route identity

Date frozen: 2026-08-31 (Asia/Shanghai), before any full-demand route is built.

The admitted demand manifest is `12832` bytes with SHA-256
`835fa16ccc50395505c95fa2c894321b80069f2829348952e9fb3e4b0e87cff1`.
It accounts for all 1,718,313 published TNP/taxi rows: 1,422,405 have two
publicly locatable endpoints and are expanded, while 295,908 remain explicitly
classified as public-location-suppressed or outside the 77-area city scope.

Routing uses one SUMO 1.22 `duarouter` invocation for all seven offset days so
the 4.49-million-edge network and contraction hierarchy are loaded once. The
frozen settings are `routing-algorithm=CH`, `bulk-routing=true`, eight routing
threads, and seed `5307`. Route repair and ignored errors remain disabled. Any
unroutable trip rejects the package.

The routed output must contain exactly one vehicle for every included input ID.
For each vehicle, the global departure and first/last route edges must equal the
input trip. The combined output is split by identity into seven files, each day
offset is subtracted, and local departures must remain sorted inside
`[0,86400)`. Per-day and per-source counts must equal the demand manifest.

The seven strict SUMO configurations use seeds `5201` through `5207`, one-second
steps, a 108,000-second completion cap, demand scale one,
`max-depart-delay=-1`, fatal route errors, zero time-based teleportation,
junction collision checks, and `collision.action=warn`. No route, safety, or
efficacy outcome is yet available under this identity.
