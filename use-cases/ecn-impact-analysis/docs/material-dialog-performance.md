# Material overlay responsiveness

Clicking a material number opens its native dialog immediately with the requested
identifier and **Loading material details…**. Complete details replace this state
after the authenticated material API responds. Loading and errors contain no
price, BOM, acceptance, or drawing evidence presented as successfully loaded.
An error offers **Retry** for the same material and acceptance task.

The material session owns a child abort signal. Closing, Escape, material
replacement, and route teardown cancel client requests and dispose drawing
resources. Late successes and failures cannot overwrite or reopen the overlay.
The Close control remains mounted through loading, failure, and completion;
keyboard focus stays there if the user has moved to it. Otherwise focus moves
to the loaded title or the material's required acceptance action. Closing returns
focus to the original table control, including after counterpart navigation.

The backend still validates exact ECN membership first. Active price publication,
material price rows, and latest completed drawing report now share one short
database connection. This avoids repeated pool checkout checks and connection
cleanup between these reads. Connection health checks remain enabled. Mutable
prices and reports are read on each opening and are not cached with immutable
assessment evidence. A failed read still fails the complete response.

Local measurements before the change showed about 2.08 seconds from click to
opening, with about 2.06 seconds in the material API and 22 milliseconds in
rendering. With immediate opening, the same local material opened in about
11 milliseconds; data readiness still depended on the API. These development
measurements are illustrative, not production service-level guarantees.
Two comparisons of the same three live database reads measured 2.13/2.34 seconds
with separate connections and 1.57/1.78 seconds with one shared connection,
saving about 0.56 seconds in each comparison. These exclude cold snapshot
construction and whole-request overhead.
A final warm full-request check opened the overlay in 3 milliseconds and showed
complete details in 1.40 seconds. After development-server restarts, first reads
took 35–39 seconds while the assessment cache was cold. Cold assessment
reconstruction and network latency remain separate costs; this change does not
claim to eliminate them. Client cancellation stops fetch delivery and ignores
late results; already-running synchronous database work still finishes safely
under the existing backend cancellation policy.

Regression coverage holds API responses until explicitly released to verify
immediate desktop/mobile opening, accessible loading, retry, focus retention,
and closing during a read. Unit coverage additionally checks request replacement,
route cleanup, late responses, one-connection operational reads, refreshed
publications/reports, and membership rejection before any operational lookup.
