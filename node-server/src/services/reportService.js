import { uploadBuffer } from "./cloudinaryService.js";
import prisma from "../config/db.js";
import { assertInspectionAccess } from "./dataScopingService.js";

/**
 * Creates an INSPECTION_SUMMARY Report record for an inspection the caller
 * is scoped to see. The returned report is the shared inspection summary
 * consumed by Controller, Reviewer, and Inspector report views.
 */
export async function createInspectionSummaryReport(user, inspectionId) {
  const inspection = await prisma.inspection.findUnique({
    where: { id: inspectionId },
    include: {
      product: { select: { brandName: true, commodityName: true, category: true } },
      inspector: { select: { id: true, fullName: true, badgeNumber: true, district: true, state: true } },
      violations: { select: { ruleCode: true, severity: true, title: true } },
      reports: { select: { id: true, fileUrl: true, reportType: true } },
    },
  });

  if (!inspection) {
    return null;
  }

  // Row-level security: caller must be within jurisdictional scope.
  assertInspectionAccess(user, inspection);

  // Reuse the certificate persisted at scan time (or a previous generation):
  // fetching the stored URL is instant, while re-uploading on every click is
  // slow and can serve stale annotated imagery. Pre-PDF certificates (legacy
  // .html uploads) are regenerated so downloads are always real PDFs.
  const existingCertificate = inspection.reports.find((r) => r.fileUrl);
  if (existingCertificate?.fileUrl?.endsWith(".pdf")) {
    return existingCertificate;
  }

  // Build the statutory PDF, upload once to Cloudinary, and persist the
  // fileUrl on the Report row for all future fetches.
  const generated = await generateAndUploadReport({
    inspection,
    product: inspection.product,
    violations: inspection.violations,
    declarations: inspection.extractedDeclarations || [],
    inspector: inspection.inspector,
    category: inspection.product?.category || "general",
  });

  if (generated?.report) {
    // Migrate a legacy HTML-certificate row in place instead of stacking a
    // duplicate report per click.
    if (existingCertificate) {
      const migrated = await prisma.report.update({
        where: { id: existingCertificate.id },
        data: { fileUrl: generated.fileUrl, reportType: "STATUTORY_COMPLIANCE_REPORT" },
      });
      return migrated;
    }
    return generated.report;
  }

  // Cloudinary upload failed — still persist the summary record so the
  // action is auditable; the UI degrades to a summary-only preview.
  const report = await prisma.report.create({
    data: {
      inspectionId: inspection.id,
      generatedById: user.id,
      reportType: "INSPECTION_SUMMARY",
      content: {
        inspection_id: inspection.id,
        product_name:
          inspection.product?.brandName ||
          inspection.product?.commodityName ||
          "Pre-Packaged Consumer Commodity",
        category: inspection.product?.category || "general",
        compliance_score: inspection.complianceScore,
        status: inspection.status,
        inspector: inspection.inspector?.fullName || null,
        violations_count: inspection.violations.length,
        major_violations: inspection.violations.filter((v) => v.severity === "MAJOR").length,
        top_rule_codes: [...new Set(inspection.violations.map((v) => v.ruleCode))].slice(0, 5),
        generated_by: user.fullName || user.email,
        generated_at: new Date().toISOString(),
      },
      fileUrl: null,
    },
  });

  return report;
}

/**
 * Builds an official standalone HTML Statutory Compliance Audit Certificate.
 */
export function buildReportHtml({
  inspection,
  product = null,
  violations = [],
  declarations = [],
  inspector = null,
  category = "general",
}) {
  const isCompliant = inspection.status === "COMPLIANT" || inspection.status === "compliant";
  const score = Math.round(inspection.complianceScore ?? 100);
  const createdDate = inspection.createdAt
    ? new Date(inspection.createdAt).toLocaleString("en-IN", {
        dateStyle: "full",
        timeStyle: "medium",
      })
    : new Date().toLocaleString("en-IN");

  const productName =
    product?.brandName && product?.commodityName
      ? `${product.brandName} - ${product.commodityName}`
      : product?.commodityName ||
        product?.brandName ||
        "Pre-Packaged Consumer Commodity";

  const catDisplay = (category || product?.category || "general").toUpperCase();
  const certId = `ALMAC-${inspection.id.slice(0, 8).toUpperCase()}`;

  const rowsHtml = (declarations || [])
    .map((d) => {
      const fieldName = (d.field_name || d.id || "Declaration").replace(/_/g, " ").toUpperCase();
      const text = d.extracted_text || d.parsed_value || "Detected";
      const statusClass = d.status === "FAIL" || d.is_violation ? "badge-fail" : "badge-pass";
      const statusText = d.status === "FAIL" || d.is_violation ? "NON-COMPLIANT" : "COMPLIANT";
      return `
        <tr>
          <td style="font-weight: 600; color: #1e293b;">${fieldName}</td>
          <td style="font-family: monospace; color: #0f172a;">${text}</td>
          <td><span class="badge ${statusClass}">${statusText}</span></td>
        </tr>
      `;
    })
    .join("");

  const violationsHtml = (violations || []).length
    ? (violations || [])
        .map((v, i) => {
          const ruleCode = v.ruleCode || v.rule_id || "RULE_VIOLATION";
          const title = v.title || `${v.field_name || "Statutory"} Violation`;
          const desc = v.description || "";
          const sev = v.severity || "MAJOR";
          const citation =
            typeof v.citation === "object" && v.citation?.source_document
              ? `${v.citation.source_document} - ${v.citation.section_title || ""}`
              : typeof v.citation === "string"
              ? v.citation
              : "Legal Metrology (Packaged Commodities) Rules, 2011";

          return `
          <div class="violation-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
              <span style="font-weight: 700; color: #991b1b;">#${i + 1} ${title}</span>
              <span class="badge badge-sev-${sev.toLowerCase()}">${sev}</span>
            </div>
            <p style="margin: 4px 0; font-size: 13px; color: #334155;">${desc}</p>
            <div style="font-size: 12px; color: #64748b; margin-top: 4px;">
              <strong>Legal Citation:</strong> <em>${citation}</em>
            </div>
          </div>
        `;
        })
        .join("")
    : `<div style="padding: 16px; background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 8px; color: #065f46; font-weight: 600;">
        Zero statutory non-compliances detected. Packaging adheres 100% to mandatory Legal Metrology declarations.
       </div>`;

  const evidenceImg = inspection.annotatedImagePath || inspection.imagePath;

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Statutory Compliance Audit Certificate - ${certId}</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 32px;
      background: #f8fafc;
      color: #0f172a;
      line-height: 1.5;
    }
    .container {
      max-width: 860px;
      margin: 0 auto;
      background: #ffffff;
      padding: 40px;
      border-radius: 12px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
      border: 1px solid #e2e8f0;
    }
    .header {
      text-align: center;
      border-bottom: 2px solid #0f172a;
      padding-bottom: 20px;
      margin-bottom: 24px;
    }
    .header h1 {
      margin: 0;
      font-size: 22px;
      letter-spacing: 0.5px;
      color: #0f172a;
      text-transform: uppercase;
    }
    .header h2 {
      margin: 6px 0 0 0;
      font-size: 14px;
      color: #475569;
      font-weight: 500;
    }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px 24px;
      background: #f1f5f9;
      padding: 16px;
      border-radius: 8px;
      margin-bottom: 24px;
      font-size: 13px;
    }
    .meta-item strong {
      color: #475569;
      display: inline-block;
      width: 140px;
    }
    .verdict-banner {
      padding: 16px 20px;
      border-radius: 8px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .verdict-pass {
      background: #ecfdf5;
      border: 1px solid #10b981;
      color: #065f46;
    }
    .verdict-fail {
      background: #fef2f2;
      border: 1px solid #ef4444;
      color: #991b1b;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 28px;
      font-size: 13px;
    }
    th, td {
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid #e2e8f0;
    }
    th {
      background: #f8fafc;
      color: #475569;
      font-weight: 600;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.5px;
    }
    .badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
    }
    .badge-pass { background: #d1fae5; color: #065f46; }
    .badge-fail { background: #fee2e2; color: #991b1b; }
    .badge-sev-critical { background: #7f1d1d; color: #ffffff; }
    .badge-sev-major { background: #f87171; color: #ffffff; }
    .badge-sev-minor { background: #fef08a; color: #854d0e; }
    .violation-card {
      background: #fff5f5;
      border: 1px solid #fecaca;
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 12px;
    }
    .evidence-section {
      margin-top: 28px;
      border-top: 1px solid #e2e8f0;
      padding-top: 20px;
    }
    .evidence-img {
      max-width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid #cbd5e1;
      margin-top: 8px;
    }
    .footer {
      margin-top: 36px;
      border-top: 1px dashed #cbd5e1;
      padding-top: 16px;
      text-align: center;
      font-size: 11px;
      color: #64748b;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Government of India &middot; Legal Metrology Division</h1>
      <h2>Automated Statutory Packaging Compliance Audit Certificate</h2>
    </div>

    <div class="meta-grid">
      <div class="meta-item"><strong>Certificate No:</strong> ${certId}</div>
      <div class="meta-item"><strong>Date of Audit:</strong> ${createdDate}</div>
      <div class="meta-item"><strong>Product Name:</strong> ${productName}</div>
      <div class="meta-item"><strong>Category:</strong> ${catDisplay}</div>
      <div class="meta-item"><strong>Inspection ID:</strong> <code>${inspection.id}</code></div>
      <div class="meta-item"><strong>Audited By:</strong> ${inspector?.fullName || "Automated AI Inspector"}</div>
    </div>

    <div class="verdict-banner ${isCompliant ? "verdict-pass" : "verdict-fail"}">
      <div>
        <div style="font-size: 16px; font-weight: 800;">
          ${isCompliant ? "LEGAL METROLOGY COMPLIANT" : "STATUTORY VIOLATION NOTICE ISSUED"}
        </div>
        <div style="font-size: 13px; margin-top: 4px;">
          ${isCompliant ? "Packaging adheres to all audited mandatory declarations under PCR 2011." : "Non-compliances detected under Legal Metrology (Packaged Commodities) Rules, 2011."}
        </div>
      </div>
      <div style="text-align: right;">
        <div style="font-size: 28px; font-weight: 800;">${score}%</div>
        <div style="font-size: 11px; text-transform: uppercase;">Compliance Index</div>
      </div>
    </div>

    <h3 style="font-size: 15px; text-transform: uppercase; color: #0f172a; margin-bottom: 12px;">Mandatory Declarations Audit</h3>
    <table>
      <thead>
        <tr>
          <th>Statutory Declaration</th>
          <th>Detected Value / Package Text</th>
          <th>Audit Status</th>
        </tr>
      </thead>
      <tbody>
        ${rowsHtml || '<tr><td colspan="3" style="text-align: center; color: #64748b;">No declarations recorded</td></tr>'}
      </tbody>
    </table>

    <h3 style="font-size: 15px; text-transform: uppercase; color: #0f172a; margin-bottom: 12px;">Detected Statutory Violations</h3>
    ${violationsHtml}

    ${evidenceImg ? `
      <div class="evidence-section">
        <h3 style="font-size: 15px; text-transform: uppercase; color: #0f172a; margin-bottom: 8px;">Audit Evidence Exhibit</h3>
        <p style="font-size: 12px; color: #64748b; margin-top: 0;">Bounding-box annotated packaging artifact preserved in secure evidence storage.</p>
        <img src="${evidenceImg}" alt="Audit Evidence" class="evidence-img" />
      </div>
    ` : ""}

    <div class="footer">
      This is a system-generated audit report produced by ALMAC LabelLens Compliance Engine.
    </div>
  </div>
</body>
</html>`;
}

/**
 * Renders a statutory report as a real, paginated PDF (pdfkit) — no browser
 * or print dialog involved, so structure and page breaks are exact.
 * Returns a Promise<Buffer>.
 */
export async function buildReportPdf({
  inspection,
  product = null,
  violations = [],
  declarations = [],
  inspector = null,
  category = "general",
}) {
  const { default: PDFDocument } = await import("pdfkit");

  // Standard PDF fonts cannot encode ₹ or non-Latin glyphs — normalize so
  // text never fails to encode.
  const pdfSafe = (value) =>
    String(value ?? "")
      .replace(/₹\s?/g, "Rs. ")
      .replace(/[^\x20-\x7E\u00A0-\u00FF\n]/g, "")
      .trim();

  const isCompliant = inspection.status === "COMPLIANT" || inspection.status === "compliant";
  const score = Math.round(inspection.complianceScore ?? 100);
  const certId = `ALMAC-${String(inspection.id).slice(0, 8).toUpperCase()}`;
  const createdDate = inspection.createdAt
    ? new Date(inspection.createdAt).toLocaleString("en-IN", { dateStyle: "full", timeStyle: "medium" })
    : new Date().toLocaleString("en-IN");
  const productName =
    product?.brandName && product?.commodityName
      ? `${product.brandName} - ${product.commodityName}`
      : product?.commodityName || product?.brandName || "Pre-Packaged Consumer Commodity";
  const catDisplay = (category || product?.category || "general").toUpperCase();

  const doc = new PDFDocument({ size: "A4", margin: 40, info: { Title: `Statutory Compliance Audit Report ${certId}` } });
  const chunks = [];
  doc.on("data", (chunk) => chunks.push(chunk));
  const done = new Promise((resolve, reject) => {
    doc.on("end", () => resolve(Buffer.concat(chunks)));
    doc.on("error", reject);
  });

  const M = 40;
  const W = doc.page.width - M * 2;
  const BOTTOM = doc.page.height - 60;
  const ensureSpace = (needed) => {
    if (doc.y + needed > BOTTOM) doc.addPage();
  };

  // --- Header ---------------------------------------------------------------
  doc.font("Helvetica-Bold").fontSize(13).fillColor("#0f172a")
    .text("GOVERNMENT OF INDIA · LEGAL METROLOGY DIVISION", M, M, { width: W, align: "center" });
  doc.font("Helvetica").fontSize(10).fillColor("#475569")
    .text("Automated Statutory Packaging Compliance Audit Report", M, doc.y + 4, { width: W, align: "center" });
  doc.moveTo(M, doc.y + 8).lineTo(M + W, doc.y + 8).lineWidth(1.5).strokeColor("#0f172a").stroke();
  doc.y += 20;

  // --- Meta grid -------------------------------------------------------------
  const metaRows = [
    ["Certificate No:", certId],
    ["Date of Audit:", createdDate],
    ["Product Name:", productName],
    ["Category:", catDisplay],
    ["Inspection ID:", String(inspection.id)],
    ["Audited By:", inspector?.fullName || "Automated AI Inspector"],
  ];
  const metaRowHeight = 16;
  const metaBoxH = metaRows.length * metaRowHeight + 16;
  ensureSpace(metaBoxH);
  const metaTop = doc.y;
  doc.rect(M, metaTop, W, metaBoxH).fill("#f1f5f9");
  doc.font("Helvetica").fontSize(9);
  metaRows.forEach(([label, value], i) => {
    const y = metaTop + 8 + i * metaRowHeight;
    doc.fillColor("#475569").font("Helvetica-Bold").text(label, M + 10, y, { continued: false });
    doc.fillColor("#0f172a").font("Helvetica").text(pdfSafe(value), M + 120, y, { width: W - 130 });
  });
  doc.y = metaTop + metaBoxH + 14;

  // --- Verdict banner ---------------------------------------------------------
  ensureSpace(56);
  const bannerColor = isCompliant ? "#ecfdf5" : "#fef2f2";
  const bannerBorder = isCompliant ? "#10b981" : "#ef4444";
  const bannerText = isCompliant ? "#065f46" : "#991b1b";
  doc.rect(M, doc.y, W, 46).fill(bannerColor);
  doc.rect(M, doc.y, W, 46).lineWidth(1).stroke(bannerBorder);
  doc.fillColor(bannerText).font("Helvetica-Bold").fontSize(11)
    .text(
      isCompliant ? "LEGAL METROLOGY COMPLIANT" : "STATUTORY VIOLATION NOTICE ISSUED",
      M + 12, doc.y + 9
    );
  doc.font("Helvetica").fontSize(8).fillColor(bannerText)
    .text(
      isCompliant
        ? "Packaging adheres to audited mandatory declarations under PCR 2011."
        : "Non-compliances detected under Legal Metrology (Packaged Commodities) Rules, 2011.",
      M + 12, doc.y + 24
    );
  doc.font("Helvetica-Bold").fontSize(18).fillColor(bannerText)
    .text(`${score}%`, M + W - 90, doc.y + 10, { width: 78, align: "right" });
  doc.font("Helvetica").fontSize(7).fillColor("#64748b")
    .text("COMPLIANCE INDEX", M + W - 90, doc.y + 32, { width: 78, align: "right" });
  doc.y += 58;

  // --- Section: declarations --------------------------------------------------
  const sectionTitle = (title) => {
    ensureSpace(30);
    doc.font("Helvetica-Bold").fontSize(10).fillColor("#0f172a")
      .text(title.toUpperCase(), M, doc.y);
    doc.y += 6;
  };

  sectionTitle("Mandatory Declarations Audit");
  const colName = M, colValue = M + 210, colStatus = M + W - 90;
  const colNameW = 200, colValueW = W - 300;
  if (!declarations.length) {
    doc.font("Helvetica").fontSize(9).fillColor("#64748b")
      .text("No declarations recorded", M, doc.y);
    doc.y += 20;
  } else {
    declarations.forEach((d, i) => {
      const name = pdfSafe((d.field_name || d.id || "Declaration").replace(/_/g, " ").toUpperCase());
      const value = pdfSafe(d.extracted_text || d.parsed_value || "Detected");
      const failed = d.status === "FAIL" || d.is_violation;
      const rowH = Math.max(
        18,
        doc.heightOfString(name, { width: colNameW }) + 8,
        doc.heightOfString(value, { width: colValueW }) + 8
      );
      ensureSpace(rowH + 4);
      if (i % 2 === 0) doc.rect(M, doc.y, W, rowH).fill("#f8fafc");
      doc.fillColor("#1e293b").font("Helvetica-Bold").fontSize(8.5)
        .text(name, colName, doc.y + 4, { width: colNameW });
      doc.fillColor("#0f172a").font("Courier").fontSize(8.5)
        .text(value, colValue, doc.y + 4, { width: colValueW });
      doc.font("Helvetica-Bold").fontSize(8).fillColor(failed ? "#991b1b" : "#065f46")
        .text(failed ? "NON-COMPLIANT" : "COMPLIANT", colStatus, doc.y + 4, { width: 86, align: "right" });
      doc.y += rowH;
      doc.moveTo(M, doc.y).lineTo(M + W, doc.y).lineWidth(0.5).strokeColor("#e2e8f0").stroke();
    });
  }
  doc.y += 14;

  // --- Section: violations -----------------------------------------------------
  sectionTitle("Detected Statutory Violations");
  if (!violations.length) {
    ensureSpace(30);
    doc.rect(M, doc.y, W, 26).fill("#ecfdf5");
    doc.fillColor("#065f46").font("Helvetica-Bold").fontSize(9)
      .text("Zero statutory non-compliances detected. Packaging adheres 100% to mandatory declarations.",
        M + 10, doc.y + 8, { width: W - 20 });
    doc.y += 38;
  } else {
    violations.forEach((v, i) => {
      const ruleCode = pdfSafe(v.ruleCode || v.rule_id || "RULE_VIOLATION");
      const title = pdfSafe(v.title || `${v.field_name || "Statutory"} Violation`);
      const desc = pdfSafe(v.description || "");
      const sev = String(v.severity || "MAJOR").toUpperCase();
      const titleW = W - 70;
      const descH = desc ? doc.heightOfString(desc, { width: W - 24 }) : 0;
      const cardH = 22 + descH + 14;
      ensureSpace(cardH + 8);
      doc.roundedRect(M, doc.y, W, cardH, 4).fill("#fff5f5");
      doc.roundedRect(M, doc.y, W, cardH, 4).lineWidth(1).stroke("#fecaca");
      doc.fillColor("#991b1b").font("Helvetica-Bold").fontSize(9)
        .text(`#${i + 1} ${title}`, M + 10, doc.y + 7, { width: titleW });
      doc.font("Helvetica-Bold").fontSize(8)
        .text(sev, M + W - 55, doc.y + 7, { width: 45, align: "right" });
      if (desc) {
        doc.fillColor("#334155").font("Helvetica").fontSize(8.5)
          .text(desc, M + 10, doc.y + 22, { width: W - 20 });
      }
      doc.y += cardH + 8;
    });
  }
  doc.y += 8;

  // --- Evidence image -----------------------------------------------------------
  const evidenceImg = inspection.annotatedImagePath || inspection.imagePath;
  if (evidenceImg && /^https?:\/\//.test(evidenceImg)) {
    try {
      const imgRes = await fetch(evidenceImg, { signal: AbortSignal.timeout(15_000) });
      const type = imgRes.headers.get("content-type") || "";
      if (imgRes.ok && type.startsWith("image/")) {
        const buf = Buffer.from(await imgRes.arrayBuffer());
        ensureSpace(240);
        sectionTitle("Audit Evidence Exhibit");
        doc.font("Helvetica").fontSize(8).fillColor("#64748b")
          .text("Bounding-box annotated packaging artifact preserved in secure evidence storage.", M, doc.y);
        doc.y += 14;
        doc.image(buf, M, doc.y, { fit: [W, 320], align: "center" });
        doc.y = Math.min(doc.y + 330, BOTTOM);
      }
    } catch {
      // Evidence image is best-effort — the textual record stands alone.
    }
  }

  // --- Footer ---------------------------------------------------------------------
  ensureSpace(50);
  doc.moveTo(M, doc.y + 6).lineTo(M + W, doc.y + 6).lineWidth(0.75).strokeColor("#cbd5e1").stroke();
  doc.font("Helvetica").fontSize(8).fillColor("#64748b")
    .text(
      "This is a system-generated audit report produced by ALMAC LabelLens Compliance Engine.",
      M, doc.y + 14, { width: W, align: "center" }
    );

  doc.end();
  return done;
}

/**
 * Generates an official PDF report, uploads it to Cloudinary, and saves it in
 * the NeonDB reports table (fileUrl points at the real PDF for direct
 * download — no print dialog involved).
 */
export async function generateAndUploadReport({
  inspection,
  product = null,
  violations = [],
  declarations = [],
  inspector = null,
  category = "general",
}) {
  try {
    const pdfBuffer = await buildReportPdf({
      inspection,
      product,
      violations,
      declarations,
      inspector,
      category,
    });

    const filename = `report_${inspection.id}.pdf`;

    // Upload to Cloudinary with resource_type: "raw"
    const uploadRes = await uploadBuffer(pdfBuffer, {
      folder: "labellens/reports",
      filename,
      resourceType: "raw",
    });

    const secureUrl = uploadRes?.secure_url;
    if (!secureUrl) {
      throw new Error("Cloudinary upload did not return a secure_url");
    }

    // Persist into NeonDB reports table
    const report = await prisma.report.create({
      data: {
        inspectionId: inspection.id,
        generatedById: inspection.inspectorId || null,
        reportType: "STATUTORY_COMPLIANCE_REPORT",
        content: {
          scan_id: inspection.id,
          product_name: product?.brandName || product?.commodityName || "Pre-Packaged Consumer Commodity",
          category: product?.category || category || "general",
          compliance_score: inspection.complianceScore,
          status: inspection.status,
          violations_count: (violations || []).length,
          declarations_count: (declarations || []).length,
        },
        fileUrl: secureUrl,
      },
    });

    return {
      report,
      fileUrl: secureUrl,
    };
  } catch (error) {
    console.error(`Failed to generate and upload report for inspection ${inspection?.id}:`, error);
    return null;
  }
}
