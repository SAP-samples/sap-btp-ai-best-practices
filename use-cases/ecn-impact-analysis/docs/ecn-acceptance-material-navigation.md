# Materials blocking ECN acceptance

**BOM Change Evidence** includes a **Blocked status** column. Missing purchasing
entries show **Blocked** with the reason beneath it. Materials
with purchasing evidence show **Not blocked**; server-exempt materials and
other removed or unchanged parts show **Not required**.
New parts awaiting the acceptance plan show **Checking…** or **Not assessed**
when status is unavailable. The overall reason remains in the BOM section and
**Acceptance rules**. The extra explanation bar is removed from the footer.

Blockers without a corresponding BOM row appear under **Materials without BOM
placement**, with unknown placement stated explicitly. A blocker repeated in
multiple assemblies has the same status and required action in every row.

The blocked material's number opens its existing
details. A highlighted **Required before ECN acceptance** section appears below its
identifier, with affected plants, known assemblies, the material master action,
and source-update instructions. Closing the material returns focus to its
button in the table. The overlay opens immediately with the requested material
number and a loading status, then shows the complete API response. Closing it,
switching materials, or leaving the page cancels the material session, including
drawing reads and analysis. Superseded responses cannot overwrite or reopen it.
Material reads that fail show an inline error and **Retry** in the overlay;
blocked-material actions also retain their error beside the material number.
The table contains its own horizontal scrolling on small
screens and keeps its scroll position when review state updates.

**Open Purchase Info Records** takes the reviewer directly to that source card
in Settings. The current application publishes replacement workbooks rather
than editing individual purchasing records. Add the missing entries to the
workbook, publish the replacement, and import the ECN again to assess it with
the updated source. Existing assessments retain their pinned source versions.

Blocker identities come from the server acceptance plan's `missing_price` and
`parts` entries with `price: MISSING`. This uses the existing purchasing-source
membership rule; it does not infer blockers from the message text or change
acceptance requirements. Source-backed context comes from the same ECN snapshot,
including materials without a BOM row. The new flow does not save a review or
change SAP data.
