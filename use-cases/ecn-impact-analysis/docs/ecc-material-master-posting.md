# Writing an accepted ECN into SAP ECC (design, simulated today)

## What this is

When an engineer accepts an ECN in the Engineering review card, the ECN's
consequences must end up in SAP ECC. This document explains how that write
would work once it is built.

Today the write is **simulated**. Accepting records the decision in HANA, and
the card shows "SAP updated (simulated)". Nothing is sent to SAP.

## The process

| Step | Who | What happens | Built today |
|---|---|---|---|
| 1. Review | Engineer | Sees one line per part: the material master change and whether the part has a purchase price | Yes |
| 2. Decide | Engineer | **Accept ECN**, or **Reject** with a comment | Yes |
| 3. Write to SAP | System | Every line becomes one SAP call (see below) | No, simulated |
| 4. Result | System | Each line shows done or failed, with SAP's own message | No, simulated as done |

The engineer can accept an ECN only when **every new part has a purchase price**.
A part has a price when its material number appears in the purchasing info
record dataset (any plant). Old parts that are phased out need no price.
Otherwise the ECN is blocked, and the only action is to send it back to
engineering. See [engineering-review.md](engineering-review.md).

## Which SAP calls would be made

All material master changes go through one BAPI, `BAPI_MATERIAL_SAVEDATA`
(create and change material master data, listed on SAP Help under
"Material Master BAPIs"). Each call must be followed by
`BAPI_TRANSACTION_COMMIT` when `RETURN` has no error, or
`BAPI_TRANSACTION_ROLLBACK` when it does. Make one call per material and plant,
so one failed line cannot leave the other lines half done.

| Line in the card | BAPI input | ECC field |
|---|---|---|
| New part: release in plant | `HEADDATA-MATERIAL`, `PLANTDATA-PLANT`, `PLANTDATA-PUR_STATUS`, `PLANTDATA-PVALIDFROM` | MARC-MMSTA / MARC-MMSTD |
| New part not yet set up in plant: create and release | Same as above, plus the view flags in `HEADDATA` that extend the part to the plant | MARC |
| Old part: phase out in plant | `PLANTDATA-PUR_STATUS`, `PLANTDATA-PVALIDFROM` | MARC-MMSTA / MARC-MMSTD |
| Old part: phase out in all plants | `CLIENTDATA-PUR_STATUS`, `CLIENTDATA-PVALIDFROM` | MARA-MSTAE / MARA-MSTDE |

Every filled structure also needs its `...X` twin (`PLANTDATAX`,
`CLIENTDATAX`) with `'X'` on each field being changed. Otherwise SAP ignores
the value.

The field names above are the standard ECC names. Check them against the
client system before building the call.

### The two status columns

The material master file has two "MS" columns:

| Column | Meaning | ECC field |
|---|---|---|
| 1st MS | Engineering status, cross-plant (for example PROTO, APP FOR SERIES) | MARA-MSTAE, sent as `CLIENTDATA-PUR_STATUS` |
| 2nd MS | Plant status (for example PROTO, PHASE IN, SERIAL, PHASED OUT) | MARC-MMSTA, sent as `PLANTDATA-PUR_STATUS` |

The file holds numeric codes (`20`, `30`, `40`, `50`, `70`, `90`), and the codes
differ per plant. The code list lives in ECC table **T141 / T141T** (material
status and its texts). The real write needs that list, for example from a
one-time SE16 export, to turn "SERIAL in PL10" into a code. Until then the card
describes changes in words only.

### Purchasing info records

A purchasing info record is a separate object, not the material master. In
ECC it is normally created with function module `ME_INFORECORD_MAINTAIN` (or
`ME_INFORECORD_MAINTAIN_MULTI`); check which one is released on the client
system. It needs a vendor, a purchasing organization and a **price**.

The data has no price. Because the write is simulated, this design takes a
shortcut and **creates no info records**. A new part counts as "has a price"
when it already appears in the purchasing info record dataset. A new part that
does not appear there blocks the ECN.

## How the app would reach ECC

The client runs ECC on-premise. The app runs on SAP BTP Cloud Foundry. The
path between them is:

```
App (Cloud Foundry)
  -> BTP Destination service      (where: ECC host, client, user, auth)
  -> BTP Connectivity service     (tunnel endpoint inside BTP)
  -> SAP Cloud Connector          (installed in the client network, opens the tunnel outwards)
  -> SAP ECC                      (RFC or HTTP)
```

- **SAP Cloud Connector** is installed once in the client network, next to
  ECC. It opens an outbound tunnel to the BTP subaccount, so no inbound
  firewall port is needed. Its access control lists exactly which ECC hosts and
  which RFC function modules or HTTP paths BTP may call. Allow only
  `BAPI_MATERIAL_SAVEDATA`, `BAPI_TRANSACTION_COMMIT`,
  `BAPI_TRANSACTION_ROLLBACK` and, if used, the info record function.
- **Destination service** holds the connection to ECC, with proxy type
  `OnPremise`. The app reads it at run time, so no ECC credentials are stored
  in the app.
- **Connectivity service** gives the app the proxy that carries the call
  through the tunnel.

There are two ways to make the call itself:

| Option | How | Fit for this app |
|---|---|---|
| A. HTTP (SOAP) | Expose the BAPI as a web service in ECC transaction SOAMANAGER, then call it over HTTP through the Connectivity proxy | Good. The Python app only needs an HTTP client |
| B. RFC through SAP Integration Suite | An iFlow with the RFC receiver adapter calls the BAPI; the app calls the iFlow over HTTPS | Good. Needs an Integration Suite tenant |
| C. RFC directly from the app | `pyrfc` and the SAP NW RFC SDK inside the app | Poor. The Cloud Foundry Python buildpack does not ship the RFC SDK |

Option A or B is recommended. The decision belongs to the client's integration
team.

## What the client must confirm before the write is built

1. The status code list from T141 / T141T, per plant.
2. Which SAP call path to use (option A or B above).
3. The technical user for the Destination, and its SAP authorizations
   (authorization object M_MATE_* for material master changes).
4. Whether info records should ever be created from this app.

## Related files

| Area | File |
|---|---|
| Lines shown before acceptance, and the block rule | `api/app/ecn/evidence/acceptance_plan.py` |
| Server refuses a blocked acceptance | `api/app/ecn/services/reviews.py` (`record_review`) |
| Review card | `ui/src/components/engineering-review.js` |
