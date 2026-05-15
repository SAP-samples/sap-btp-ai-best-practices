"use strict";

function parseCSV(text) {
  text = text.replace(/^﻿/, "");
  const rows = [];
  let headers = null;
  let pos = 0;
  const len = text.length;

  function parseField() {
    if (pos < len && text[pos] === '"') {
      pos++;
      let field = "";
      while (pos < len) {
        if (text[pos] === '"') {
          pos++;
          if (pos < len && text[pos] === '"') { field += '"'; pos++; }
          else break;
        } else {
          field += text[pos++];
        }
      }
      return field;
    } else {
      let field = "";
      while (pos < len && text[pos] !== "," && text[pos] !== "\n" && text[pos] !== "\r") {
        field += text[pos++];
      }
      return field.trim();
    }
  }

  function parseLine() {
    const fields = [];
    while (pos < len && text[pos] !== "\n" && text[pos] !== "\r") {
      fields.push(parseField());
      if (pos < len && text[pos] === ",") pos++;
    }
    if (pos < len && text[pos] === "\r") pos++;
    if (pos < len && text[pos] === "\n") pos++;
    return fields;
  }

  headers = parseLine().map(function (h) { return h.trim(); });

  while (pos < len) {
    if (text[pos] === "\r" || text[pos] === "\n") {
      if (text[pos] === "\r") pos++;
      if (pos < len && text[pos] === "\n") pos++;
      continue;
    }
    const fields = parseLine();
    if (fields.length === 0 || (fields.length === 1 && fields[0] === "")) continue;
    const row = {};
    for (let i = 0; i < headers.length; i++) {
      row[headers[i]] = fields[i] !== undefined ? fields[i] : "";
    }
    rows.push(row);
  }

  return { headers, rows };
}

function parseAmount(s) {
  if (s === null || s === undefined || s === "") return NaN;
  return parseFloat(String(s).replace(/,(?=\d)/g, "").trim());
}

function amountKey(n) {
  return Math.round(n * 100);
}

self.onmessage = function (e) {
  const msg = e.data;
  if (msg.type !== "MATCH") return;

  try {
    const cfg = msg.config;

    self.postMessage({ type: "STATUS", text: "Parsing invoices..." });
    const invoiceParsed = parseCSV(msg.invoiceCSV);
    const invoices = invoiceParsed.rows;

    self.postMessage({ type: "STATUS", text: "Parsing payments..." });
    const paymentParsed = parseCSV(msg.paymentCSV);
    const payments = paymentParsed.rows;

    if (invoices.length > 0) {
      const fi = invoices[0];
      if (!(cfg.invoiceNumberCol in fi))
        throw new Error('Invoice column "' + cfg.invoiceNumberCol + '" not found. Available: ' + Object.keys(fi).join(", "));
      if (!(cfg.invoiceAmountCol in fi))
        throw new Error('Invoice amount column "' + cfg.invoiceAmountCol + '" not found. Available: ' + Object.keys(fi).join(", "));
    }
    if (payments.length > 0) {
      const fp = payments[0];
      if (!(cfg.paymentAmountCol in fp))
        throw new Error('Payment amount column "' + cfg.paymentAmountCol + '" not found. Available: ' + Object.keys(fp).join(", "));
      for (let k = 0; k < cfg.paymentTextCols.length; k++) {
        if (!(cfg.paymentTextCols[k] in fp))
          throw new Error('Payment text column "' + cfg.paymentTextCols[k] + '" not found. Available: ' + Object.keys(fp).join(", "));
      }
    }

    self.postMessage({ type: "STATUS", text: "Indexing payments by amount..." });
    const paymentIndex = new Map();
    const toleranceCents = Math.round(cfg.tolerance * 100);

    for (let i = 0; i < payments.length; i++) {
      const p = payments[i];
      const payAmt = parseAmount(p[cfg.paymentAmountCol]) / 100;
      if (isNaN(payAmt)) continue;

      const key = amountKey(payAmt);
      let text = "";
      for (let k = 0; k < cfg.paymentTextCols.length; k++) {
        text += " " + (p[cfg.paymentTextCols[k]] || "");
      }

      const entry = { payAmt, text, bankRef: (p[cfg.paymentRefCol] || "") };
      if (!paymentIndex.has(key)) paymentIndex.set(key, []);
      paymentIndex.get(key).push(entry);
    }

    self.postMessage({ type: "STATUS", text: "Matching invoices to payments..." });

    const tolerance = cfg.tolerance;
    const results = [];
    let matched = 0;

    for (let i = 0; i < invoices.length; i++) {
      const inv = invoices[i];
      const invNo = (inv[cfg.invoiceNumberCol] || "").trim();
      const invAmt = parseAmount(inv[cfg.invoiceAmountCol]);

      let matchStatus = "unmatched";
      let matchedPayAmt = null;
      let matchedBankRef = "";
      let matchCount = 0;

      if (!isNaN(invAmt) && invNo !== "") {
        const invCents = amountKey(invAmt);
        const minKey = invCents - toleranceCents;
        const maxKey = invCents + toleranceCents;

        const amountCandidates = [];
        for (let ck = minKey; ck <= maxKey; ck++) {
          const bucket = paymentIndex.get(ck);
          if (bucket) {
            for (let j = 0; j < bucket.length; j++) {
              if (Math.abs(invAmt - bucket[j].payAmt) <= tolerance) {
                amountCandidates.push(bucket[j]);
              }
            }
          }
        }

        const textMatches = [];
        for (let j = 0; j < amountCandidates.length; j++) {
          if (amountCandidates[j].text.indexOf(invNo) !== -1) {
            textMatches.push(amountCandidates[j]);
          }
        }

        matchCount = textMatches.length;
        if (matchCount === 1) {
          matchStatus = "matched";
          matchedPayAmt = textMatches[0].payAmt;
          matchedBankRef = textMatches[0].bankRef;
          matched++;
        }
      }

      results.push({
        invoiceNumber: invNo,
        invoiceAmount: isNaN(invAmt) ? inv[cfg.invoiceAmountCol] : invAmt,
        paymentAmount: matchedPayAmt !== null ? matchedPayAmt : "",
        bankRef: matchedBankRef,
        matchCount,
        matchStatus,
        matched: matchStatus === "matched",
      });

      if (i > 0 && i % 1000 === 0) {
        self.postMessage({ type: "PROGRESS", percent: Math.round((i / invoices.length) * 100) });
      }
    }

    const total = invoices.length;
    const stats = { total, matched, unmatched: total - matched };
    self.postMessage({ type: "RESULT", rows: results, stats });
  } catch (err) {
    self.postMessage({ type: "ERROR", message: err.message });
  }
};
