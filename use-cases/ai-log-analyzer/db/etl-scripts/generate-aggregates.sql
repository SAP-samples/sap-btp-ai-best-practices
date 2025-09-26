
-- create new table for messages and their calculated processing dates
DROP TABLE #MessageProcessingDates
CREATE LOCAL TEMPORARY TABLE #MessageProcessingDates (
    ProcessingDate DATE,
    messageID INTEGER,
    MSGID VARCHAR(50),
    MSGNO VARCHAR(50)
);


-- Insert data into the temporary table
INSERT INTO #MessageProcessingDates
WITH ConvertedMessages AS (
  SELECT
    ID,
    MSGID,
    MSGNO,
    TO_TIMESTAMP(SUBSTRING(CREATIONTSTAMP, 1, 14), 'YYYYMMDDHH24MISS') AS ts
    -- TO_TIMESTAMP(SUBSTRING(CREATIONTSTAMP, 1, 27), 'YYYY-MM-DD HH24:MI:SS.FF7') AS ts
  FROM SAP_BTP_AI_MESSAGES
  WHERE CREATIONTSTAMP IS NOT NULL
)
SELECT
  CASE
    WHEN TO_TIME(ts) >= '18:00:00'
      THEN ADD_DAYS(CAST(ts AS DATE), 1)
    ELSE CAST(ts AS DATE)
  END AS ProcessingDate,
  ID AS messageID,
  MSGID,
  MSGNO
FROM ConvertedMessages;

select count(*) from #MessageProcessingDates

-- SELECT mpd.ProcessingDate, mpd.MSGID, mpd.MSGNO, ex.messageshorttext, count(mpd.messageID) from #MessageProcessingDates mpd
-- join SAP_BTP_AI_SITUATIONS ex on ex.messageClass = mpd.MSGID and ex.MSGNO = mpd.MSGNO
-- group by mpd.ProcessingDate, mpd.MSGID, mpd.MSGNO, ex.messageshorttext order by mpd.processingdate, mpd.msgno


DELETe FROM  SAP_BTP_AI_MESSAGES


-- populate MessageRunAggregate

-- DELETE FROM SAP_BTP_AI_MESSAGETIMEAGGREGATES

INSERT INTO SAP_BTP_AI_MESSAGETIMEAGGREGATES
    (ID, DATE, SITUATION_ID, MESSAGECOUNT)
SELECT
    -- Build TIMEAGGREGATE ID: ProcessingDate in YYYYMMDD format + '-' + MSGID + '-' + MSGNO
    TO_VARCHAR(mpd.ProcessingDate, 'YYYYMMDD') || '-' || mpd.MSGID || '-' || mpd.MSGNO AS  ID,
    TO_VARCHAR(mpd.ProcessingDate, 'YYYYMMDD') AS DATE,
    ex.ID AS SITUATION_ID,
    COUNT(*) AS MESSAGECOUNT
FROM #MessageProcessingDates mpd
JOIN SAP_BTP_AI_SITUATIONS ex
    ON ex.messageClass = mpd.MSGID
   AND ex.MSGNO = mpd.MSGNO
GROUP BY
    mpd.ProcessingDate,
    mpd.MSGID,
    mpd.MSGNO,
    ex.ID;
    
    
-- UPDATE Messages with  timeAggregate_ID

UPDATE SAP_BTP_AI_MESSAGES
SET 
    timeAggregate_ID = (
        SELECT TO_VARCHAR(mpd.ProcessingDate, 'YYYYMMDD') || '-' || mpd.MSGID || '-' || mpd.MSGNO
        FROM #MessageProcessingDates mpd
        WHERE SAP_BTP_AI_MESSAGES.ID = mpd.messageID
    )
WHERE 
    EXISTS (
        SELECT 1 
        FROM #MessageProcessingDates mpd
        WHERE SAP_BTP_AI_MESSAGES.ID = mpd.messageID
    );


-- generate fake data for a couple more days...

/* 
    i want to read the existing messages and copy them to previous days. to make it easier to modify, the ID PK for each batch / day should start with 100000 * batch + existing ID. 
    I have a db table with ~40000 entries and I want to create mock data by copying the existing entries for the previous couple of days.
*/

INSERT INTO "FF7D82B9D8284945AB85C0584FDB52E4"."SAP_BTP_AI_MESSAGES" (
  "CREATEDAT",
  "CREATEDBY",
  "MODIFIEDAT",
  "MODIFIEDBY",
  "ID",
  "MANDT",
  "MSGHDL",
  "PROBCLASS",
  "MSGTYPE",
  "MSGID",
  "MSGNO",
  "MSGV1",
  "MSGV2",
  "MSGV3",
  "MSGV4",
  "IS_ASSIGNED",
  "CREATIONTSTAMP",
  "USR",
  "TCODE",
  "PROG",
  "MOD",
  "DELETIONTSTAMP",
  "STATUS",
  "DELETION_INDICATOR",
  "RELCONT",
  "MSGPSTAMP",
  "RUNDATE",
  "PRODUCT_ID",
  "LOCATION_ID",
  "LOCATIONFROM_ID",
  "PRODUCT_HIERARCHY_ID",
  "LOCATION_HIERARCHY_ID",
  "OFFER_ID",
  "VENDOR_FUND_ID",
  "KPRM",
  "MODEL_ID",
  "FORECAST_ID",
  "CONTROLLER_ID",
  "JOB_ID",
  "DIAG_ID",
  "SALES_ORG_ID",
  "DISTR_CHNL_ID",
  "ORDER_CHNL_ID",
  "INBOUND_CONTEXT",
  "COUNTER",
  "ALLO_PLAN_ID",
  "COLOR",
  "MARKET_UNIT_ID",
  "ALLO_WL_ID",
  "REFERENCE_ID",
  "ORDERPLANITEMUUID",
  "REPL_RUN_ID",
  "ORDERPLANITEM4AUUID",
  "TIMEAGGREGATE_ID"
)
SELECT
  -- shift the timestamp back by dayOffset days
  ADD_DAYS(
    TO_TIMESTAMP(SUBSTR(M."CREATIONTSTAMP",1,14), 'YYYYMMDDHH24MISS'),
    -B.dayOffset
  ) AS "CREATEDAT",

  M."CREATEDBY",

  ADD_DAYS(
    TO_TIMESTAMP(SUBSTR(M."CREATIONTSTAMP",1,14), 'YYYYMMDDHH24MISS'),
    -B.dayOffset
  ) AS "MODIFIEDAT",

  M."MODIFIEDBY",

  -- new PK: 100000 * batch + old ID
  (100000 * B.batch + M."ID") AS "ID",

  M."MANDT",
  M."MSGHDL",
  M."PROBCLASS",
  M."MSGTYPE",
  M."MSGID",
  M."MSGNO",
  M."MSGV1",
  M."MSGV2",
  M."MSGV3",
  M."MSGV4",
  M."IS_ASSIGNED",

  -- rebuild your NVARCHAR timestamp: new YYYYMMDD + original time/fraction
  TO_NVARCHAR(
    ADD_DAYS(TO_DATE(SUBSTR(M."CREATIONTSTAMP",1,8), 'YYYYMMDD'), -B.dayOffset),
    'YYYYMMDD'
  ) || SUBSTR(M."CREATIONTSTAMP",9) AS "CREATIONTSTAMP",

  M."USR",
  M."TCODE",
  M."PROG",
  M."MOD",
  M."DELETIONTSTAMP",
  M."STATUS",
  M."DELETION_INDICATOR",
  M."RELCONT",
  M."MSGPSTAMP",
  M."RUNDATE",
  M."PRODUCT_ID",
  M."LOCATION_ID",
  M."LOCATIONFROM_ID",
  M."PRODUCT_HIERARCHY_ID",
  M."LOCATION_HIERARCHY_ID",
  M."OFFER_ID",
  M."VENDOR_FUND_ID",
  M."KPRM",
  M."MODEL_ID",
  M."FORECAST_ID",
  M."CONTROLLER_ID",
  M."JOB_ID",
  M."DIAG_ID",
  M."SALES_ORG_ID",
  M."DISTR_CHNL_ID",
  M."ORDER_CHNL_ID",
  M."INBOUND_CONTEXT",
  M."COUNTER",
  M."ALLO_PLAN_ID",
  M."COLOR",
  M."MARKET_UNIT_ID",
  M."ALLO_WL_ID",
  M."REFERENCE_ID",
  M."ORDERPLANITEMUUID",
  M."REPL_RUN_ID",
  M."ORDERPLANITEM4AUUID",
  M."TIMEAGGREGATE_ID"
FROM "FF7D82B9D8284945AB85C0584FDB52E4"."SAP_BTP_AI_MESSAGES" AS M

-- define your batches: batch=1 → yesterday, batch=2 → day before
CROSS JOIN (
  SELECT 1 AS batch, 1 AS dayOffset FROM DUMMY
  UNION ALL
  SELECT 2 AS batch, 2 AS dayOffset FROM DUMMY
  UNION ALL
  SELECT 3 AS batch, 3 AS dayOffset FROM DUMMY
  UNION ALL
  SELECT 4 AS batch, 4 AS dayOffset FROM DUMMY
  UNION ALL
  SELECT 5 AS batch, 5 AS dayOffset FROM DUMMY
) AS B;

----

-- ... add some variance 

DELETE
  FROM "SAP_BTP_AI_MESSAGES"
WHERE
  RAND() < 0.2;                 -- delete ~20% at random

SELECT count(*) from SAP_BTP_AI_MESSAGES




select * from SAP_BTP_AI_MESSAGETIMEAGGREGATES

select * from SAP_BTP_AI_SITUATIONS

delete from SAP_BTP_AI_ASSESSMENT