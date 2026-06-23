const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, LevelFormat,
  TableOfContents, TabStopType, TabStopPosition
} = require('docx');
const fs = require('fs');

// ─── Helpers ───────────────────────────────────────────────────────────────
const CONTENT_WIDTH = 9026; // A4 with 1-inch margins (DXA)

const border = { style: BorderStyle.SINGLE, size: 4, color: "2E4057" };
const borders = { top: border, bottom: border, left: border, right: border };

const headerBorder = { style: BorderStyle.SINGLE, size: 6, color: "2E75B6" };
const lightBorder = { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" };
const lightBorders = {
  top: lightBorder, bottom: lightBorder, left: lightBorder, right: lightBorder
};

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, bold: true, size: 32, font: "Times New Roman", color: "1F3864" })],
    spacing: { before: 400, after: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "2E75B6", space: 1 } }
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, bold: true, size: 28, font: "Times New Roman", color: "1F4E79" })],
    spacing: { before: 300, after: 160 }
  });
}

function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, bold: true, size: 26, font: "Times New Roman", color: "2E4057" })],
    spacing: { before: 240, after: 120 }
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({
      text,
      size: opts.size || 24,
      font: opts.font || "Times New Roman",
      bold: opts.bold || false,
      italics: opts.italic || false,
      color: opts.color || "000000"
    })],
    alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { before: opts.before || 100, after: opts.after || 120, line: 360 },
    indent: opts.indent ? { firstLine: 720 } : undefined
  });
}

function paraRuns(runs, opts = {}) {
  return new Paragraph({
    children: runs.map(r => new TextRun({
      text: r.text,
      bold: r.bold || false,
      italics: r.italic || false,
      size: r.size || 24,
      font: r.font || "Times New Roman",
      color: r.color || "000000"
    })),
    alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { before: opts.before || 100, after: opts.after || 120, line: 360 },
    indent: opts.indent ? { firstLine: 720 } : undefined
  });
}

function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    children: [new TextRun({ text, size: 24, font: "Times New Roman" })],
    spacing: { before: 80, after: 80, line: 340 }
  });
}

function numbered(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "numbers", level },
    children: [new TextRun({ text, size: 24, font: "Times New Roman" })],
    spacing: { before: 80, after: 80, line: 340 }
  });
}

function mathFormula(text) {
  return new Paragraph({
    children: [new TextRun({ text, size: 24, font: "Courier New", color: "1A237E" })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 160, after: 160 },
    shading: { fill: "F5F5FF", type: ShadingType.CLEAR }
  });
}

function code(text) {
  return new Paragraph({
    children: [new TextRun({ text, size: 20, font: "Courier New", color: "1A237E" })],
    alignment: AlignmentType.LEFT,
    spacing: { before: 60, after: 60, line: 300 },
    shading: { fill: "F0F0F8", type: ShadingType.CLEAR }
  });
}

function emptyLine() {
  return new Paragraph({ children: [new TextRun({ text: "", size: 24 })], spacing: { before: 80, after: 80 } });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, size: 22, font: "Times New Roman", italic: true, color: "444444" })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 200 }
  });
}

function makeTable(headers, rows, colWidths) {
  const totalW = colWidths.reduce((a, b) => a + b, 0);
  const hdrBg = "D6E4F0";
  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h, i) => new TableCell({
          borders: lightBorders,
          width: { size: colWidths[i], type: WidthType.DXA },
          shading: { fill: hdrBg, type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          verticalAlign: VerticalAlign.CENTER,
          children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, size: 22, font: "Times New Roman" })], alignment: AlignmentType.CENTER })]
        }))
      }),
      ...rows.map((row, ri) => new TableRow({
        children: row.map((cell, ci) => new TableCell({
          borders: lightBorders,
          width: { size: colWidths[ci], type: WidthType.DXA },
          shading: { fill: ri % 2 === 0 ? "FFFFFF" : "F7FBFF", type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [new Paragraph({ children: [new TextRun({ text: cell, size: 22, font: "Times New Roman" })], alignment: AlignmentType.LEFT })]
        }))
      }))
    ]
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// DOCUMENT CONTENT
// ═══════════════════════════════════════════════════════════════════════════

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1080, hanging: 360 } } } }
        ]
      },
      {
        reference: "numbers",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } }
        ]
      }
    ]
  },
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Times New Roman", color: "1F3864" },
        paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Times New Roman", color: "1F4E79" },
        paragraph: { spacing: { before: 300, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Times New Roman", color: "2E4057" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 2 } }
    ]
  },
  sections: [
    // ─── TRANG BÌA ─────────────────────────────────────────────────────────
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 }
        }
      },
      children: [
        new Paragraph({ children: [new TextRun({ text: "TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN", bold: true, size: 26, font: "Times New Roman", color: "1F3864" })], alignment: AlignmentType.CENTER, spacing: { before: 0, after: 60 } }),
        new Paragraph({ children: [new TextRun({ text: "ĐẠI HỌC QUỐC GIA TP.HCM", bold: true, size: 26, font: "Times New Roman", color: "1F3864" })], alignment: AlignmentType.CENTER, spacing: { before: 0, after: 60 } }),
        new Paragraph({ children: [new TextRun({ text: "KHOA CÔNG NGHỆ THÔNG TIN", bold: true, size: 26, font: "Times New Roman", color: "1F3864" })], alignment: AlignmentType.CENTER, spacing: { before: 0, after: 400 } }),
        new Paragraph({ children: [new TextRun({ text: "─────────────────────────────────────────", size: 24, color: "2E75B6" })], alignment: AlignmentType.CENTER, spacing: { before: 0, after: 400 } }),
        new Paragraph({ children: [new TextRun({ text: "KHÓA LUẬN TỐT NGHIỆP", bold: true, size: 40, font: "Times New Roman", color: "1F3864" })], alignment: AlignmentType.CENTER, spacing: { before: 200, after: 200 } }),
        new Paragraph({ children: [new TextRun({ text: "ĐỀ TÀI", bold: true, size: 28, font: "Times New Roman", color: "333333" })], alignment: AlignmentType.CENTER, spacing: { before: 200, after: 200 } }),
        new Paragraph({ children: [new TextRun({ text: "ATTENTION-BASED TOKEN MERGING", bold: true, size: 36, font: "Times New Roman", color: "C00000" })], alignment: AlignmentType.CENTER, spacing: { before: 100, after: 60 } }),
        new Paragraph({ children: [new TextRun({ text: "ASPECT BASED SENTIMENT ANALYSIS", bold: true, size: 36, font: "Times New Roman", color: "C00000" })], alignment: AlignmentType.CENTER, spacing: { before: 0, after: 600 } }),
        new Paragraph({ children: [new TextRun({ text: "Chuyên ngành: Khoa học Máy tính", size: 26, font: "Times New Roman" })], alignment: AlignmentType.CENTER, spacing: { before: 0, after: 80 } }),
        new Paragraph({ children: [new TextRun({ text: "Mã ngành: 7480101", size: 26, font: "Times New Roman" })], alignment: AlignmentType.CENTER, spacing: { before: 0, after: 600 } }),
        new Paragraph({ children: [new TextRun({ text: "Giảng viên hướng dẫn: [Tên giảng viên hướng dẫn]", bold: true, size: 26, font: "Times New Roman", color: "1F3864" })], alignment: AlignmentType.CENTER, spacing: { before: 0, after: 120 } }),
        new Paragraph({ children: [new TextRun({ text: "Sinh viên thực hiện: [Tên sinh viên]", bold: true, size: 26, font: "Times New Roman", color: "1F3864" })], alignment: AlignmentType.CENTER, spacing: { before: 0, after: 60 } }),
        new Paragraph({ children: [new TextRun({ text: "MSSV: [Mã số sinh viên]", size: 26, font: "Times New Roman" })], alignment: AlignmentType.CENTER, spacing: { before: 0, after: 600 } }),
        new Paragraph({ children: [new TextRun({ text: "TP. Hồ Chí Minh, năm 2025", bold: true, size: 26, font: "Times New Roman", color: "333333" })], alignment: AlignmentType.CENTER, spacing: { before: 0, after: 0 } }),
      ]
    },
    // ─── NỘI DUNG CHÍNH ────────────────────────────────────────────────────
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 }
        }
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              children: [
                new TextRun({ text: "Khóa luận tốt nghiệp", size: 20, font: "Times New Roman", italics: true, color: "555555" }),
                new TextRun({ text: "\t", size: 20 }),
                new TextRun({ text: "Attention-based Token Merging ABSA", size: 20, font: "Times New Roman", italics: true, color: "555555" })
              ],
              tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_WIDTH }],
              border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "2E75B6", space: 1 } }
            })
          ]
        })
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              children: [
                new TextRun({ text: "\t", size: 20 }),
                new PageNumber({ size: 20, font: "Times New Roman" })
              ],
              tabStops: [{ type: TabStopType.CENTER, position: CONTENT_WIDTH / 2 }],
              border: { top: { style: BorderStyle.SINGLE, size: 4, color: "2E75B6", space: 1 } }
            })
          ]
        })
      },
      children: [
        // ── LỜI CẢM ƠN ────────────────────────────────────────────────────
        heading1("LỜI CẢM ƠN"),
        para("Để hoàn thành khóa luận này, tôi xin bày tỏ lòng biết ơn sâu sắc đến:", { indent: true }),
        para("Quý thầy/cô trong Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên – Đại học Quốc gia TP.HCM đã tận tình giảng dạy và tạo nền tảng kiến thức vững chắc trong suốt quá trình học tập.", { indent: true }),
        para("Giảng viên hướng dẫn đã định hướng khoa học, góp ý chi tiết và động viên tôi trong từng giai đoạn của quá trình nghiên cứu.", { indent: true }),
        para("Gia đình và bạn bè đã luôn ủng hộ, khích lệ tôi trong suốt thời gian thực hiện đề tài.", { indent: true }),
        emptyLine(),
        para("TP. Hồ Chí Minh, năm 2025", { align: AlignmentType.RIGHT }),
        para("Sinh viên thực hiện", { align: AlignmentType.RIGHT }),
        para("[Tên sinh viên]", { align: AlignmentType.RIGHT, bold: true }),
        pageBreak(),

        // ── TÓM TẮT ───────────────────────────────────────────────────────
        heading1("TÓM TẮT"),
        para("Đề tài: Attention-based Token Merging Aspect Based Sentiment Analysis", { bold: true }),
        emptyLine(),
        para("Khóa luận nghiên cứu và xây dựng hệ thống Phân tích Cảm xúc theo Khía cạnh (Aspect-Based Sentiment Analysis – ABSA) cho tập dữ liệu đánh giá khách sạn tiếng Việt. Hệ thống giải quyết bài toán trích xuất bộ ba (aspect term, aspect category, sentiment) theo hai kiến trúc: Joint (đánh giá ATE và APC đồng thời trên nhãn gold) và Pipeline (T5 sinh aspect term rồi đưa vào APC độc lập).", { indent: true }),
        para("Hai phương pháp kỹ thuật cốt lõi được đề xuất và tích hợp: (1) Local Context Focus (LCF) với hai biến thể CDM/CDW tập trung biểu diễn vào vùng ngữ cảnh quanh khía cạnh; (2) Token Merging (ToMe) với ba chiến lược mới – bipartite, sequential local và attention-weighted – nhằm giảm dư thừa token mà không làm suy giảm chất lượng phân loại. Bộ phân loại FAST-LCF-BERT được huấn luyện đa nhiệm (sentiment + aspect category) với weighted loss để xử lý mất cân bằng nhãn.", { indent: true }),
        para("Kết quả thực nghiệm cho thấy kiến trúc Joint đạt Micro F1 bộ ba tốt nhất là 71,50% (T5, seq_resize) và 71,18% (BERT, lcf_seq_cdm_resize), vượt Pipeline (~53%) khoảng 18 điểm do lan truyền lỗi từ ATE. CDM cải thiện Macro F1 trên nhãn thiểu số thêm 6–9 điểm so với CDW. Các cấu hình ToMe không làm giảm – thậm chí cải thiện nhẹ – chất lượng so với baseline, xác nhận tiềm năng áp dụng gộp token trong ABSA.", { indent: true }),
        emptyLine(),
        para("Từ khóa: ABSA, Aspect Term Extraction, Token Merging, Local Context Focus, BERT, T5, Vietnamese hotel reviews, đa nhiệm.", { italic: true }),
        pageBreak(),

        // ── ABSTRACT ──────────────────────────────────────────────────────
        heading1("ABSTRACT"),
        para("Title: Attention-based Token Merging Aspect Based Sentiment Analysis", { bold: true }),
        emptyLine(),
        para("This thesis proposes and evaluates a system for Aspect-Based Sentiment Analysis (ABSA) on Vietnamese hotel review data. The system tackles the joint triplet extraction problem – predicting (aspect term, aspect category, sentiment) tuples – under two architectures: Joint (ATE and APC evaluated simultaneously with gold labels) and Pipeline (T5 generates aspect terms which are then passed to APC independently).", { indent: true }),
        para("Two core technical methods are integrated: (1) Local Context Focus (LCF) with CDM/CDW variants to focus representations on the aspect's local context; (2) Token Merging (ToMe) with three strategies – bipartite, sequential local, and attention-weighted – to reduce token redundancy without degrading classification quality. A multi-task FAST-LCF-BERT classifier is trained with weighted cross-entropy loss to handle class imbalance.", { indent: true }),
        para("Experimental results show the Joint architecture achieves the best triplet Micro F1 of 71.50% (T5, seq_resize) and 71.18% (BERT, lcf_seq_cdm_resize), outperforming Pipeline (~53%) by approximately 18 points due to error propagation from the ATE stage. CDM improves Macro F1 on minority labels by 6–9 points over CDW. ToMe configurations maintain or slightly improve quality compared to the baseline, confirming the viability of token merging for ABSA.", { indent: true }),
        emptyLine(),
        para("Keywords: ABSA, Aspect Term Extraction, Token Merging, Local Context Focus, BERT, T5, Vietnamese hotel reviews, multi-task learning.", { italic: true }),
        pageBreak(),

        // ── MỤC LỤC ──────────────────────────────────────────────────────
        heading1("MỤC LỤC"),
        new TableOfContents("Mục lục", {
          hyperlink: true,
          headingStyleRange: "1-3",
          stylesWithLevels: [
            { styleId: "Heading1", level: 1 },
            { styleId: "Heading2", level: 2 },
            { styleId: "Heading3", level: 3 }
          ]
        }),
        pageBreak(),

        // ═══════════════════════════════════════════════════════════════════
        // CHƯƠNG I - GIỚI THIỆU
        // ═══════════════════════════════════════════════════════════════════
        heading1("CHƯƠNG I. GIỚI THIỆU"),

        heading2("1.1. Đặt vấn đề"),
        para("Cùng với sự phát triển bùng nổ của các nền tảng đặt phòng trực tuyến và mạng xã hội du lịch, khối lượng đánh giá (review) của khách hàng dành cho các cơ sở lưu trú tăng trưởng nhanh chóng cả về số lượng lẫn mức độ chi tiết. Các đánh giá này chứa đựng thông tin phản hồi có giá trị cao giúp doanh nghiệp nhận diện điểm mạnh, điểm yếu trong vận hành và cải thiện chất lượng dịch vụ.", { indent: true }),
        para("Tuy nhiên, một đánh giá duy nhất thường đề cập đồng thời đến nhiều khía cạnh khác nhau với cảm xúc trái chiều. Ví dụ câu: \"The room was beautiful but the staff was rude\" thể hiện cảm xúc tích cực đối với khía cạnh phòng (FACILITY) nhưng lại tiêu cực đối với khía cạnh nhân viên (SERVICE). Các hệ thống phân tích cảm xúc truyền thống chỉ gán một nhãn cảm xúc duy nhất cho toàn bộ câu, không phản ánh được các mâu thuẫn cảm xúc cục bộ như vậy.", { indent: true }),
        para("Phân tích cảm xúc theo khía cạnh (Aspect-Based Sentiment Analysis – ABSA) ra đời để giải quyết hạn chế này. ABSA xác định từng khía cạnh được đề cập và phân loại cảm xúc tương ứng với từng khía cạnh đó, cung cấp thông tin chi tiết và chính xác hơn nhiều so với phân tích cảm xúc mức câu.", { indent: true }),
        para("Sự xuất hiện của các mô hình ngôn ngữ tiền huấn luyện dựa trên Transformer như BERT (Devlin et al., 2019) đã cải thiện đáng kể độ chính xác của ABSA. Đặc biệt, kiến trúc FAST-LCF-BERT kết hợp cơ chế Local Context Focus (LCF) cho phép mô hình tập trung biểu diễn vào vùng ngữ cảnh xung quanh khía cạnh đang xét. Tuy nhiên, chi phí tính toán của self-attention O(L²) là rào cản đáng kể trong triển khai thực tế quy mô lớn.", { indent: true }),
        para("Token Merging (ToMe – Bolya et al., CVPR 2023) là kỹ thuật gần đây giúp giảm độ dài chuỗi bằng cách gộp các token tương đồng, ban đầu được đề xuất cho Vision Transformer. Việc áp dụng ToMe vào ABSA tiềm ẩn rủi ro: nếu các token thuộc cụm từ khía cạnh bị gộp không cẩn thận, thông tin cục bộ quan trọng nhất cho phân loại cảm xúc có thể bị mất.", { indent: true }),

        heading2("1.2. Mục tiêu nghiên cứu"),
        para("Khóa luận hướng đến bốn mục tiêu cụ thể:", { indent: true }),
        numbered("Xây dựng hệ thống ABSA đầu-cuối cho tập dữ liệu đánh giá khách sạn tiếng Việt, trích xuất bộ ba (aspect term, aspect category, sentiment) theo hai kiến trúc Joint và Pipeline."),
        numbered("Đề xuất và cài đặt ba chiến lược Token Merging mới (bipartite, sequential local, attention-weighted) thích nghi cho bài toán ABSA, bảo vệ token khía cạnh trong quá trình gộp."),
        numbered("Đánh giá thực nghiệm hệ thống tích hợp LCF (CDM/CDW) và ToMe trên bộ dữ liệu thực tế, so sánh 12 cấu hình với hai backbone BERT và T5."),
        numbered("Phân tích ảnh hưởng của từng thành phần (LCF, ToMe, chiến lược gộp, biến thể CDM/CDW) lên chất lượng biểu diễn và độ chính xác phân loại."),
        emptyLine(),

        heading2("1.3. Phạm vi và giới hạn"),
        para("Nghiên cứu tập trung vào dữ liệu đánh giá khách sạn tiếng Việt (2.448 mẫu train, 304 dev, 312 test) với 6 nhóm khía cạnh: AMENITY, BRANDING, EXPERIENCE, FACILITY, LOYALTY, SERVICE và 3 nhãn cảm xúc: Positive, Negative, Neutral. Toàn bộ thực nghiệm sử dụng chế độ resize (nội suy chuỗi sau gộp trở lại độ dài gốc) nhằm cô lập câu hỏi chất lượng biểu diễn; lợi ích tăng tốc của compact mode được để lại cho hướng phát triển.", { indent: true }),

        heading2("1.4. Đóng góp của khóa luận"),
        bullet("Hệ thống ABSA đầu-cuối cho tiếng Việt với kiến trúc Joint và Pipeline, tích hợp T5 cho ATE và FAST-LCF-BERT đa nhiệm cho APC."),
        bullet("Ba chiến lược Token Merging mới cho chuỗi 1D văn bản: bipartite (ToMe gốc), sequential local (gộp lân cận tuần tự), attention-weighted (ưu tiên token ít quan trọng nhất)."),
        bullet("Phân tích thực nghiệm hệ thống so sánh 12 cấu hình với hai backbone trên dữ liệu khách sạn tiếng Việt, cung cấp bằng chứng thực nghiệm về hiệu quả của gộp token trong ABSA."),
        bullet("Mã nguồn và bộ dữ liệu được công bố tại: https://github.com/hotuyen21pt/KLTN-Token-Merging."),
        emptyLine(),

        heading2("1.5. Cấu trúc khóa luận"),
        para("Khóa luận được tổ chức thành năm chương:", { indent: true }),
        bullet("Chương I: Giới thiệu – đặt vấn đề, mục tiêu, phạm vi và đóng góp."),
        bullet("Chương II: Cơ sở lý thuyết – Transformer, BERT, T5, LCF, Token Merging."),
        bullet("Chương III: Phương pháp thực hiện – kiến trúc hệ thống, tích hợp LCF và ToMe, huấn luyện đa nhiệm."),
        bullet("Chương IV: Kết quả thực nghiệm – thiết lập, kết quả, phân tích và phân tích lỗi."),
        bullet("Chương V: Kết luận và hướng phát triển."),
        pageBreak(),

        // ═══════════════════════════════════════════════════════════════════
        // CHƯƠNG II - CƠ SỞ LÝ THUYẾT
        // ═══════════════════════════════════════════════════════════════════
        heading1("CHƯƠNG II. CƠ SỞ LÝ THUYẾT"),

        heading2("2.1. Bài toán Phân tích Cảm xúc theo Khía cạnh (ABSA)"),
        heading3("2.1.1. Định nghĩa bài toán"),
        para("Phân tích cảm xúc truyền thống gán một nhãn cảm xúc duy nhất cho toàn câu. Cách tiếp cận này bộc lộ hạn chế khi một câu chứa nhiều khía cạnh với cảm xúc khác nhau. ABSA (Aspect-Based Sentiment Analysis) giải quyết hạn chế này bằng cách xác định từng khía cạnh và phân loại cảm xúc tương ứng.", { indent: true }),
        para("Trong phạm vi khóa luận, bài toán được định nghĩa là trích xuất bộ ba: cho câu đánh giá x, tìm tập hợp T = {(a_i, c_i, s_i)} trong đó a_i là aspect term (cụm từ khía cạnh), c_i là aspect category, và s_i là cực tính cảm xúc. Một bộ ba được tính đúng khi và chỉ khi cả ba thành phần khớp nhãn gold đồng thời.", { indent: true }),
        emptyLine(),
        makeTable(
          ["Bài toán", "Tên đầy đủ", "Đầu vào", "Đầu ra"],
          [
            ["ATE", "Aspect Term Extraction", "Câu", "Danh sách cụm từ khía cạnh"],
            ["APC", "Aspect Polarity Classification", "Câu + khía cạnh", "Nhãn cảm xúc + nhóm khía cạnh"],
            ["ASTE", "Aspect Sentiment Triplet Extraction", "Câu", "(khía cạnh, ý kiến, cảm xúc)"],
            ["ASQP", "Aspect-Category-Opinion-Sentiment Quad", "Câu", "(khía cạnh, danh mục, ý kiến, cảm xúc)"]
          ],
          [1800, 3000, 1600, 2626]
        ),
        caption("Bảng 2.1. Các bài toán con của ABSA"),

        heading3("2.1.2. Không gian nhãn trong khóa luận"),
        para("Hệ thống sử dụng 6 nhóm khía cạnh và 3 cực tính cảm xúc:", { indent: true }),
        makeTable(
          ["Thành phần", "Tập giá trị"],
          [
            ["aspect_term", "Cụm từ xuất hiện trong câu (ví dụ: 'phòng', 'nhân viên', 'bể bơi')"],
            ["aspect_category", "AMENITY · BRANDING · EXPERIENCE · FACILITY · LOYALTY · SERVICE"],
            ["sentiment", "Positive · Negative · Neutral"]
          ],
          [2000, 7026]
        ),
        caption("Bảng 2.2. Không gian nhãn của hệ thống ABSA"),
        emptyLine(),

        heading2("2.2. Kiến trúc Transformer và Self-Attention"),
        heading3("2.2.1. Tổng quan Transformer"),
        para("Transformer (Vaswani et al., 2017) là kiến trúc nền tảng của hầu hết các mô hình ngôn ngữ hiện đại. Khác với RNN/LSTM, Transformer xử lý toàn bộ chuỗi đầu vào đồng thời thay vì tuần tự, nhờ cơ chế self-attention cho phép mỗi token \"nhìn\" trực tiếp tới mọi token khác bất kể khoảng cách.", { indent: true }),
        para("Một khối encoder Transformer gồm hai thành phần chính: Multi-Head Self-Attention (MHSA) và Feed-Forward Network (FFN), mỗi thành phần được bao bởi kết nối tắt (residual connection) và Layer Normalization.", { indent: true }),

        heading3("2.2.2. Self-Attention và Multi-Head Self-Attention"),
        para("Với chuỗi gồm n token được biểu diễn thành ma trận X ∈ R^(n×d), self-attention chiếu X thành Query, Key, Value:", { indent: true }),
        mathFormula("Q = XW^Q,    K = XW^K,    V = XW^V"),
        para("Đầu ra attention được tính bằng tích vô hướng tỉ lệ (Scaled Dot-Product Attention):", { indent: true }),
        mathFormula("Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V"),
        para("Hệ số sqrt(d_k) tránh tích vô hướng quá lớn làm softmax bão hòa. Transformer sử dụng Multi-Head Attention song song để học nhiều loại quan hệ ngữ nghĩa/cú pháp khác nhau:", { indent: true }),
        mathFormula("MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W^O"),
        para("Đây là thành phần then chốt: chi phí tính toán O(n² × d_k) do tích QK^T tạo ma trận attention kích thước n×n. Khi n giảm (nhờ gộp token), chi phí của các lớp self-attention phía sau giảm theo bậc hai – đây là động lực cốt lõi của Token Merging.", { indent: true }),

        heading2("2.3. Mô hình ngôn ngữ tiền huấn luyện"),
        heading3("2.3.1. BERT"),
        para("BERT (Bidirectional Encoder Representations from Transformers – Devlin et al., 2019) là encoder Transformer được tiền huấn luyện theo hai mục tiêu: Masked Language Modeling (MLM) – che ngẫu nhiên 15% token và dự đoán lại dựa trên ngữ cảnh hai chiều; và Next Sentence Prediction (NSP). BERT sử dụng tokenizer WordPiece, thêm token [CLS] ở đầu (biểu diễn tổng hợp cho phân loại) và [SEP] để phân tách.", { indent: true }),
        para("Đầu ra của BERT là ma trận hidden states H ∈ R^(L×d_h) với d_h = 768 (bert-base-uncased). Hai token [CLS] và [SEP] được bảo vệ tuyệt đối (protect_cls=True, protect_sep=True) khỏi bị gộp trong module Token Merging.", { indent: true }),

        heading3("2.3.2. T5"),
        para("T5 (Text-to-Text Transfer Transformer – Raffel et al., 2020) là mô hình Transformer encoder-decoder, biểu diễn mọi bài toán NLP dưới dạng sinh văn bản (text-to-text). T5 được tiền huấn luyện trên tập C4 với mục tiêu span corruption.", { indent: true }),
        para("Trong khóa luận, T5 được dùng cho bài toán ATE. ATE được hình thức hóa thành bài toán sinh có điều kiện: mô hình T5AspectExtractor nhận câu đánh giá và sinh danh sách cụm từ khía cạnh theo định dạng GAS, ví dụ: \"(phòng); (nhân viên); (thang máy)\". Câu không chứa khía cạnh sinh ra \"none\".", { indent: true }),

        heading2("2.4. Cơ chế Local Context Focus (LCF)"),
        heading3("2.4.1. Động lực"),
        para("Trong câu \"The room was beautiful but the staff was rude.\", từ 'beautiful' chỉ liên quan đến 'room', còn 'rude' chỉ liên quan đến 'staff'. Nếu mô hình dùng biểu diễn câu như nhau cho cả hai khía cạnh, thông tin nhiễu từ khía cạnh kia có thể làm giảm độ chính xác. LCF giải quyết vấn đề này bằng cách tạo biểu diễn ngữ cảnh cục bộ – trong đó các token xa khía cạnh bị giảm trọng số hoặc loại bỏ – rồi kết hợp với biểu diễn toàn cục.", { indent: true }),

        heading3("2.4.2. Khoảng cách ngữ nghĩa tương đối (SRD)"),
        para("Đầu vào LCF là vector chỉ thị lcf_vec ∈ {0,1}^L, trong đó các vị trí sub-word thuộc cụm từ khía cạnh được gán 1. Từ vector này, mô hình tính tâm và nửa độ rộng của khía cạnh:", { indent: true }),
        mathFormula("center = (a_start + a_end) / 2,   half = floor((a_end - a_start + 1) / 2)"),
        para("Khoảng cách ngữ nghĩa tương đối SRD của token tại vị trí i:", { indent: true }),
        mathFormula("SRD_i = max(0, |i - center| - half)"),
        para("Ngưỡng α (SRD_THRESHOLD, mặc định α = 5): token có SRD_i ≤ α thuộc \"ngữ cảnh cục bộ\" của khía cạnh.", { indent: true }),

        heading3("2.4.3. Hai biến thể CDM và CDW"),
        para("Context Dynamic Mask (CDM) – mặt nạ nhị phân, loại bỏ hoàn toàn token nằm ngoài phạm vi α:", { indent: true }),
        mathFormula("m_i = 1 nếu SRD_i ≤ α,   m_i = 0 nếu SRD_i > α"),
        para("Context Dynamic Weight (CDW) – trọng số giảm dần liên tục theo khoảng cách:", { indent: true }),
        mathFormula("c_i = 1 nếu SRD_i ≤ α,   c_i = (L - (SRD_i - α)) / L nếu SRD_i > α"),
        para("CDM \"tập trung cứng\" – phù hợp cho nhãn thiểu số. CDW giữ thông tin xa hơn với gradient mượt hơn – phù hợp cho nhãn phổ biến.", { indent: true }),

        heading2("2.5. Token Merging (ToMe)"),
        heading3("2.5.1. Động lực"),
        para("Chi phí self-attention tỉ lệ O(n²) theo độ dài chuỗi n. Trong ABSA, nhiều token (từ nối, dấu câu, subword lặp) mang rất ít thông tin phân biệt. Token Merging (Bolya et al., CVPR 2023) gộp các token có biểu diễn tương tự nhau thành một token đại diện bằng phép trung bình, giảm độ dài chuỗi mà không thêm tham số.", { indent: true }),

        heading3("2.5.2. Thuật toán Bipartite Soft Matching"),
        para("Thuật toán gốc của ToMe gồm các bước:", { indent: true }),
        numbered("Chia các token nội bộ (trừ [CLS], [SEP] và token khía cạnh) thành hai nhóm xen kẽ A và B."),
        numbered("Chuẩn hóa L2 vector hidden state."),
        numbered("Với mỗi a ∈ A, tìm b ∈ B có độ tương đồng cosine lớn nhất: sim(a,b) = x̂_a · x̂_b."),
        numbered("Mỗi b ∈ B chỉ được ghép tối đa một lần (one-to-one matching trên B)."),
        numbered("Hợp nhất hai token bằng trung bình: x_merged = (x_a + x_b) / 2; token b bị loại."),
        emptyLine(),

        heading3("2.5.3. Ba chiến lược gộp trong khóa luận"),
        makeTable(
          ["Chiến lược", "Cơ chế", "Đặc điểm"],
          [
            ["bipartite", "Chia token thành 2 nhóm A/B; A chọn B giống nhất (CVPR 2023)", "Gộp đồng thời, số lượng merge cố định, không phụ thuộc thứ tự"],
            ["sequential_local", "Quét trái→phải, mỗi token gộp về phía hàng xóm giống nhất; giữ ranh giới LCF", "Xếp tầng cascade, phụ thuộc thứ tự, chi phí thấp hơn"],
            ["attention_weighted", "Gộp token có attention thấp nhất vào hàng xóm cosine cao nhất; bảo vệ token attention top 25%", "Bảo toàn thông tin quan trọng nhất, cần tín hiệu attention"]
          ],
          [2200, 4000, 2826]
        ),
        caption("Bảng 2.3. Ba chiến lược Token Merging trong hệ thống"),
        emptyLine(),
        para("Cả ba chiến lược đều bảo vệ tuyệt đối [CLS], [SEP] và token thuộc vùng khía cạnh (protect_aspect=True). Khi gộp, chỉ thị khía cạnh được lan truyền qua phép lấy cực đại: lcf[src] = max(lcf[src], lcf[dst]).", { indent: true }),
        pageBreak(),

        // ═══════════════════════════════════════════════════════════════════
        // CHƯƠNG III - PHƯƠNG PHÁP THỰC HIỆN
        // ═══════════════════════════════════════════════════════════════════
        heading1("CHƯƠNG III. PHƯƠNG PHÁP THỰC HIỆN"),

        heading2("3.1. Kiến trúc tổng thể"),
        heading3("3.1.1. Bài toán và luồng xử lý"),
        para("Hệ thống giải quyết bài toán trích xuất bộ ba (Joint Triplet Extraction): từ câu đánh giá khách sạn x, trích xuất tập T = {(a_i, c_i, s_i)} với a_i là aspect term, c_i là aspect category, và s_i là cực tính. Hệ thống được tổ chức thành bốn stage nối tiếp:", { indent: true }),
        numbered("Stage 1 – Tiền xử lý: Token hóa SPC, dựng vector lcf_vec từ offset mapping."),
        numbered("Stage 2 – ATE (T5): Sinh có điều kiện danh sách aspect term + chuẩn hóa Levenshtein."),
        numbered("Stage 3 – APC (FastLcfBertMultiTask): Backbone → Token Merging → LCF → phân loại đa nhiệm."),
        numbered("Stage 4 – Tổng hợp bộ ba: Ghép (a_i, c_i, s_i)."),
        emptyLine(),

        heading3("3.1.2. Hai kiến trúc tổ chức: Joint và Pipeline"),
        makeTable(
          ["Kiến trúc", "Mô tả", "Đặc điểm chính"],
          [
            ["Joint", "APC đánh giá trên aspect term gold; ATE và APC độc lập nhau", "Cô lập năng lực bộ phân loại, không chịu lan truyền lỗi từ ATE"],
            ["Pipeline", "T5 thực hiện ATE trước, aspect term dự đoán đưa vào APC", "Linh hoạt, phản ánh hiệu suất thực tế; chịu lan truyền lỗi từ ATE"]
          ],
          [1600, 3800, 3626]
        ),
        caption("Bảng 3.1. Hai kiến trúc tổ chức bài toán ABSA đầu-cuối"),
        emptyLine(),

        heading3("3.1.3. Chi tiết Stage ATE"),
        para("Bài toán ATE được hình thức hóa thành sinh có điều kiện: mô hình T5AspectExtractor (t5-base) nhận câu và sinh danh sách cụm từ khía cạnh theo định dạng GAS:", { indent: true }),
        code("Input : \"Phòng sạch sẽ, nhân viên nhiệt tình nhưng thang máy hay hỏng.\""),
        code("Output: \"(phòng); (nhân viên); (thang máy)\""),
        para("Quá trình sinh dùng beam search (beam=4, max_len=64). Đầu ra được hậu xử lý bằng đối sánh n-gram Levenshtein: mỗi cụm từ sinh ra được ánh xạ về n-gram gần nhất thực sự xuất hiện trong câu gốc.", { indent: true }),

        heading2("3.2. Tiền xử lý và xây dựng vector LCF"),
        heading3("3.2.1. Token hóa SPC"),
        para("Mỗi mẫu được token hóa theo định dạng Sentence-Pair Classification:", { indent: true }),
        code("[CLS] <câu gốc> [SEP] <aspect term> [SEP]"),
        para("Độ dài tối đa L = 128 token. Token hóa dùng tokenizer của backbone (bert-base-uncased hoặc t5-base) với padding='max_length' và truncation=True.", { indent: true }),

        heading3("3.2.2. Xây dựng vector lcf_vec"),
        para("Vector lcf_vec ∈ {0,1}^L được dựng dựa trên offset mapping của tokenizer. Với mỗi token k trong segment A:", { indent: true }),
        bullet("Nếu khoảng ký tự của token k giao với khoảng ký tự của cụm khía cạnh → lcf_vec[k] = 1"),
        bullet("Ngược lại → lcf_vec[k] = 0"),
        bullet("Fallback: nếu không tìm thấy token nào trong segment A → dùng token_type_ids (bản sao khía cạnh ở segment B)"),
        emptyLine(),

        heading2("3.3. Kiến trúc phân loại FastLcfBertMultiTask"),
        heading3("3.3.1. Tổng quan"),
        para("Mô hình FastLcfBertMultiTask dự đoán đồng thời hai nhãn:", { indent: true }),
        bullet("Sentiment head: Linear(d_h → 3) → Softmax [positive / negative / neutral]"),
        bullet("Category head: Linear(d_h → 6) → Softmax [AMENITY / BRANDING / EXPERIENCE / FACILITY / LOYALTY / SERVICE]"),
        para("Hai đầu phân loại chia sẻ toàn bộ phần thân chung (backbone → ToMe → LCF → Fusion → pooler), chỉ khác ở lớp tuyến tính cuối.", { indent: true }),

        heading3("3.3.2. Luồng xử lý chi tiết"),
        para("Backbone BERT/T5 mã hóa đầu vào thành H ∈ R^(B×L×d_h). Sau đó:", { indent: true }),
        numbered("Token Merging (tùy chọn): Gộp token dư thừa, nội suy trở lại L."),
        numbered("Local stream: H_local = H ⊙ c (CDW) hoặc H ⊙ m (CDM) → Self-Attention (bert_SA)."),
        numbered("Global stream: H không thay đổi."),
        numbered("Fusion: H_fused = Linear_{2d→d}([H_local; H]) → Dropout → Self-Attention (bert_SA_)."),
        numbered("Pooler: lấy token [CLS] → Linear + Tanh → pooled."),
        numbered("Classification: pooled → dense_sentiment, dense_aspect_cat."),
        emptyLine(),
        mathFormula("H_fused = Linear_{2d→d}(concat(H_local, H))"),
        para("Token Merging được chèn ngay sau backbone và trước LCF. Lý do: (1) backbone đã tiền huấn luyện sẵn, tránh phá vỡ biểu diễn; (2) lcf_vec được cập nhật đồng bộ qua phép max; (3) cả hai nhánh fusion làm việc trên cùng chuỗi đã rút gọn.", { indent: true }),

        heading2("3.4. Huấn luyện đa nhiệm"),
        heading3("3.4.1. Hàm mất mát"),
        para("Hai đầu phân loại được huấn luyện đồng thời bằng tổng hai thành phần CrossEntropyLoss:", { indent: true }),
        mathFormula("L_total = L_sentiment + 1[main_mask] × L_aspect_cat"),
        para("trong đó 1[main_mask] chỉ bật trên mẫu chính (is_supplement=False). Mẫu supplement (negative.tsv, neutral.tsv) chỉ tham gia huấn luyện đầu sentiment; đầu category không học trên supplement.", { indent: true }),
        para("Cả hai thành phần sử dụng trọng số lớp (class weights) theo tần suất nghịch đảo:", { indent: true }),
        mathFormula("w_k = N / (C × n_k)"),
        para("với N là tổng số mẫu, C là số lớp, n_k là số mẫu lớp k.", { indent: true }),

        heading3("3.4.2. Chế độ resize trong Token Merging"),
        para("Toàn bộ thực nghiệm sử dụng chế độ resize=True (nội suy chuỗi sau gộp trở lại độ dài gốc L=128) vì ba lý do:", { indent: true }),
        numbered("Cô lập đúng biến cần đo: khác biệt duy nhất giữa baseline và cấu hình có gộp là nội dung biểu diễn (do trung bình token tương đồng), loại bỏ nhiễu từ độ dài chuỗi thay đổi."),
        numbered("Compact gây nhiễu so sánh: mỗi cấu hình sinh ra L' khác nhau, trộn lẫn ảnh hưởng của gộp với ảnh hưởng của độ dài chuỗi."),
        numbered("Lợi ích tốc độ của compact không đáng kể ở vị trí sau backbone: 12 lớp Transformer vẫn chạy ở L=128; compact chỉ rút ngắn đầu vào cho 2 lớp SA nhẹ phía sau."),
        emptyLine(),

        heading2("3.5. Bộ dữ liệu"),
        heading3("3.5.1. Định dạng .apc"),
        para("Dữ liệu là tập đánh giá khách sạn tiếng Việt định dạng .apc với mỗi mẫu gồm 4 dòng:", { indent: true }),
        code("$T$ rất chuyên nghiệp từ bộ phận nhà hàng đến lễ tân."),
        code("nhân viên phục vụ"),
        code("SERVICE"),
        code("Positive"),
        para("Dòng 1: câu gốc (có placeholder $T$), Dòng 2: aspect term, Dòng 3: category, Dòng 4: sentiment.", { indent: true }),

        heading3("3.5.2. Thống kê dữ liệu"),
        makeTable(
          ["Tập", "Số mẫu", "Vai trò"],
          [
            ["Train", "2.448", "Huấn luyện (kèm supplement)"],
            ["Dev", "304", "Early stopping và chọn checkpoint"],
            ["Test", "312", "Đánh giá chính thức (không dùng khi huấn luyện)"]
          ],
          [2000, 2000, 5026]
        ),
        caption("Bảng 3.2. Thống kê các tập dữ liệu"),
        emptyLine(),
        para("Dữ liệu supplement gồm negative.tsv và neutral.tsv (định dạng TSV: text / term / sentiment, không có category) được thêm vào để cân bằng nhãn cho đầu sentiment. Các mẫu này được đánh dấu is_supplement=True.", { indent: true }),
        pageBreak(),

        // ═══════════════════════════════════════════════════════════════════
        // CHƯƠNG IV - KẾT QUẢ THỰC NGHIỆM
        // ═══════════════════════════════════════════════════════════════════
        heading1("CHƯƠNG IV. KẾT QUẢ THỰC NGHIỆM"),

        heading2("4.1. Thiết lập thực nghiệm"),
        heading3("4.1.1. Siêu tham số"),
        makeTable(
          ["Siêu tham số", "Giá trị", "Ghi chú"],
          [
            ["Backbone", "bert-base-uncased / t5-base", "Encoder cho APC"],
            ["Số epoch tối đa", "15", "NUM_EPOCHS"],
            ["Early stopping", "patience = 4", "Theo dev joint F1"],
            ["Batch size", "16", "BATCH_SIZE"],
            ["Learning rate", "2 × 10⁻⁵", "AdamW optimizer"],
            ["Độ dài chuỗi", "128", "MAX_SEQ_LEN"],
            ["Dropout", "0.1", "DROPOUT"],
            ["Số đầu attention", "8", "NUM_HEADS"],
            ["SRD threshold α", "5", "Bán kính ngữ cảnh LCF"],
            ["ToMe merge steps", "2", "Số vòng gộp (post-BERT)"],
            ["Mixed precision", "Bật (AMP)", "Tự tắt nếu không có CUDA"],
            ["Random seed", "42", "Cố định để tái lập"]
          ],
          [3200, 2600, 3226]
        ),
        caption("Bảng 4.1. Siêu tham số huấn luyện module APC"),
        emptyLine(),

        heading3("4.1.2. Không gian cấu hình"),
        makeTable(
          ["Nhóm cấu hình", "LCF", "ToMe"],
          [
            ["baseline_balanced", "Không", "Không"],
            ["lcf_only_{cdm,cdw}", "Có", "Không"],
            ["{bip,seq,attn}_resize", "Không", "Có (3 chiến lược)"],
            ["lcf_{bip,seq,attn}_{cdm,cdw}_resize", "Có", "Có (3 chiến lược × CDM/CDW)"]
          ],
          [4000, 2000, 3026]
        ),
        caption("Bảng 4.2. Không gian 12 cấu hình thực nghiệm"),
        emptyLine(),

        heading3("4.1.3. Độ đo đánh giá"),
        bullet("ATE F1: khớp chính xác aspect term (Micro P/R/F1)."),
        bullet("Micro F1: tính TP/FP/FN trên toàn bộ mẫu; ưu tiên nhãn phổ biến."),
        bullet("Macro F1: trung bình F1 qua các nhãn; nhạy với nhãn thiểu số."),
        bullet("Oracle Cat/Sent/Joint Acc: độ chính xác khi cho trước aspect term gold – giới hạn trên của bộ phân loại."),
        para("Tiêu chí TP: bộ ba dự đoán (aspect, category, sentiment) khớp CHÍNH XÁC với gold – cả ba thành phần phải đúng đồng thời.", { indent: true }),
        emptyLine(),

        heading2("4.2. Kết quả thực nghiệm"),
        heading3("4.2.1. Kiến trúc Joint – backbone BERT"),
        makeTable(
          ["Cấu hình", "Micro F1 (%)", "Macro F1 (%)", "Oracle Joint (%)", "Train (s)"],
          [
            ["baseline_balanced", "70.53", "48.80", "86.54", "257"],
            ["lcf_only_cdw", "69.24", "46.66", "85.90", "265"],
            ["lcf_only_cdm", "68.92", "47.94", "84.62", "311"],
            ["bip_resize", "70.85", "49.19", "86.86", "374"],
            ["lcf_bip_cdm_resize", "69.24", "54.20", "84.94", "383"],
            ["lcf_bip_cdw_resize", "70.53", "49.61", "86.86", "463"],
            ["lcf_attn_cdw_resize", "69.57", "48.11", "84.94", "533"],
            ["lcf_seq_cdm_resize (*tốt nhất Micro*)", "71.18", "50.00", "86.86", "553"],
            ["lcf_attn_cdm_resize (*tốt nhất Macro*)", "69.89", "57.16", "86.54", "586"],
            ["lcf_seq_cdw_resize", "70.21", "50.00", "86.86", "608"],
            ["seq_resize", "69.24", "47.98", "86.22", "651"],
            ["attn_resize", "68.28", "47.76", "85.26", "684"]
          ],
          [3400, 1500, 1500, 1800, 900]
        ),
        caption("Bảng 4.3. Kết quả Joint – backbone BERT (ATE F1 = 79.87%)"),
        emptyLine(),

        heading3("4.2.2. Kiến trúc Joint – backbone T5"),
        makeTable(
          ["Cấu hình", "Micro F1 (%)", "Macro F1 (%)", "Oracle Joint (%)", "Train (s)"],
          [
            ["baseline_balanced", "69.57", "47.88", "85.90", "337"],
            ["lcf_only_cdw", "69.89", "48.76", "86.54", "345"],
            ["lcf_only_cdm", "70.21", "48.75", "86.22", "468"],
            ["bip_resize", "70.53", "48.03", "86.86", "498"],
            ["lcf_bip_cdw_resize", "70.53", "49.80", "87.18", "651"],
            ["attn_resize", "68.28", "45.48", "84.62", "659"],
            ["lcf_bip_cdm_resize (*tốt nhất Macro*)", "71.18", "56.03", "86.54", "698"],
            ["lcf_attn_cdw_resize", "70.85", "49.63", "85.58", "853"],
            ["seq_resize (*tốt nhất Micro + Oracle*)", "71.50", "50.12", "89.10", "886"],
            ["lcf_seq_cdw_resize", "69.89", "54.25", "85.90", "901"],
            ["lcf_attn_cdm_resize", "67.95", "53.40", "84.29", "913"],
            ["lcf_seq_cdm_resize", "69.89", "45.79", "85.90", "921"]
          ],
          [3400, 1500, 1500, 1800, 900]
        ),
        caption("Bảng 4.4. Kết quả Joint – backbone T5 (ATE F1 = 79.87%)"),
        emptyLine(),

        heading3("4.2.3. Kiến trúc Pipeline"),
        makeTable(
          ["Backbone", "Cấu hình", "Micro F1 (%)", "Macro F1 (%)", "Oracle Joint (%)", "Infer (s)"],
          [
            ["BERT", "baseline_balanced", "52.86", "38.41", "86.54", "10.8"],
            ["BERT", "lcf_seq_cdm_resize", "53.57", "40.29", "86.86", "14.9"],
            ["BERT", "lcf_attn_cdm_resize", "52.38", "43.90", "86.54", "14.8"],
            ["T5", "baseline_balanced", "52.86", "39.57", "85.90", "7.6"],
            ["T5", "seq_resize", "53.81", "39.98", "89.10", "14.3"],
            ["T5", "lcf_bip_cdm_resize", "53.10", "47.81", "86.54", "11.3"]
          ],
          [1200, 2800, 1300, 1300, 1500, 900]
        ),
        caption("Bảng 4.5. Kết quả Pipeline (ATE F1: BERT = 59.76%, T5 = 60.00%)"),
        emptyLine(),

        heading3("4.2.4. Kết quả GAS T5 – Mô hình single-stage"),
        para("Ngoài hai kiến trúc chính, khóa luận còn đánh giá mô hình GAS T5 sinh bộ ba trong một bước (single-stage generation):", { indent: true }),
        makeTable(
          ["Metric", "Precision (%)", "Recall (%)", "F1 (%)"],
          [
            ["Aspect Term", "79.61", "78.85", "79.23"],
            ["Category", "74.11", "73.40", "73.75"],
            ["Sentiment", "75.73", "75.00", "75.36"],
            ["Joint (all 3)", "70.87", "70.19", "70.53"]
          ],
          [2500, 2000, 2000, 2526]
        ),
        caption("Bảng 4.6. Kết quả GAS T5 trên test split (Joint F1 = 70.53%)"),
        emptyLine(),

        heading2("4.3. Phân tích và so sánh"),
        heading3("4.3.1. Joint vượt Pipeline ~18 điểm Micro F1"),
        para("Micro F1 bộ ba của Joint (~71%) cao hơn Pipeline (~53%) khoảng 18 điểm. Nguyên nhân nằm hoàn toàn ở bước ATE: ATE F1 giảm từ 79.87% (Joint) xuống ~60% (Pipeline). Bằng chứng then chốt là chỉ số Oracle Joint Acc gần như không đổi giữa hai kiến trúc với cùng cấu hình (ví dụ lcf_attn_cdm_resize: 86.54% ở cả Joint lẫn Pipeline) – tức bộ phân loại là như nhau, hạn chế đến từ chất lượng aspect term được trích xuất. Đây là biểu hiện điển hình của lan truyền lỗi trong kiến trúc pipeline.", { indent: true }),

        heading3("4.3.2. BERT và T5 tương đương về Micro F1"),
        para("Hai backbone cho Micro F1 Joint sát nhau (~71%): tốt nhất là T5 seq_resize (71.50%) và BERT lcf_seq_cdm_resize (71.18%). T5 nhỉnh hơn về Oracle Joint (89.10% so với 86.86%), cho thấy tiềm năng cao hơn nếu ATE được cải thiện. BERT huấn luyện nhanh hơn rõ rệt: baseline 257s so với 337s; các cấu hình LCF của T5 thường vượt 850-920s.", { indent: true }),

        heading3("4.3.3. CDM tốt hơn CDW về Macro F1"),
        para("Mặt nạ nhị phân CDM bảo toàn \"tập trung cứng\" tốt hơn trọng số mềm CDW, dẫn tới Macro F1 cao hơn đáng kể trên nhãn thiểu số:", { indent: true }),
        bullet("BERT, chiến lược attention_weighted: CDM đạt Macro 57.16% so với CDW 48.11% (+9.05 điểm)."),
        bullet("T5, chiến lược bipartite: CDM đạt Macro 56.03% so với CDW 49.80% (+6.23 điểm)."),
        para("Trong khi đó, Micro F1 giữa CDM và CDW chênh lệch nhỏ – CDM cải thiện chủ yếu ở nhãn hiếm mà không hy sinh nhãn phổ biến.", { indent: true }),

        heading3("4.3.4. Gộp token không làm giảm chất lượng"),
        para("Các cấu hình có ToMe không kém baseline, thậm chí tốt hơn nhẹ. Với BERT, baseline đạt Micro 70.53% còn lcf_seq_cdm_resize đạt 71.18% và bip_resize đạt 70.85%. Điều này khẳng định: việc gộp các token tương đồng (loại dư thừa) không phá hỏng biểu diễn cần cho phân loại – tiền đề quan trọng để áp dụng gộp token cho mục tiêu tăng tốc về sau.", { indent: true }),

        heading3("4.3.5. Chi phí huấn luyện trong chế độ resize"),
        para("Trong chế độ resize, các cấu hình ToMe có thời gian huấn luyện cao hơn baseline (ví dụ BERT: 257s so với 553s ở lcf_seq_cdm_resize) do thêm bước gộp và nội suy mà không rút ngắn chuỗi cho các lớp phía sau. Resize phục vụ đo chất lượng biểu diễn, không phải để tăng tốc; lợi ích tốc độ thực sự thuộc về hướng phát triển.", { indent: true }),

        heading2("4.4. Kết quả per-class: sentiment và category"),
        heading3("4.4.1. F1 theo nhãn cảm xúc"),
        makeTable(
          ["Cấu hình (BERT)", "F1-Positive (%)", "F1-Negative (%)", "F1-Neutral (%)"],
          [
            ["baseline_balanced", "76.34", "42.86", "66.67"],
            ["lcf_seq_cdm_resize", "75.97", "42.86", "85.71"],
            ["lcf_attn_cdm_resize", "75.38", "42.86", "85.71"],
            ["bip_resize", "77.78", "42.86", "66.67"]
          ],
          [3600, 2000, 2000, 1426]
        ),
        caption("Bảng 4.7. F1 theo nhãn cảm xúc – một số cấu hình BERT tiêu biểu"),
        emptyLine(),

        heading3("4.4.2. F1 theo nhóm khía cạnh"),
        makeTable(
          ["Cấu hình (BERT)", "AMENITY", "BRANDING", "EXPERIENCE", "FACILITY", "LOYALTY", "SERVICE"],
          [
            ["baseline_balanced", "76.34%", "0.00%", "61.82%", "68.93%", "76.92%", "77.19%"],
            ["lcf_seq_cdm_resize", "75.97%", "0.00%", "67.86%", "70.39%", "76.92%", "77.65%"],
            ["lcf_attn_cdm_resize", "75.38%", "0.00%", "66.67%", "66.67%", "76.92%", "77.19%"]
          ],
          [2800, 1100, 1100, 1300, 1100, 1100, 1326]
        ),
        caption("Bảng 4.8. F1 theo nhóm khía cạnh – một số cấu hình BERT tiêu biểu"),
        emptyLine(),

        heading2("4.5. Phân tích lỗi"),
        heading3("4.5.1. Lỗi ở bước trích xuất khía cạnh (ATE)"),
        para("Khảo sát results_ate_incorrect.csv cho thấy lỗi ATE tập trung ở hai dạng: (1) sai biên cụm từ – trích thừa hoặc thiếu so với gold; (2) chọn nhầm khía cạnh trong câu nhiều khía cạnh. Vì tiêu chí chấm là khớp chính xác, các lỗi biên đều bị tính sai hoàn toàn.", { indent: true }),
        makeTable(
          ["Câu (rút gọn)", "Dự đoán", "Gold"],
          [
            ["Exceptional staff and customer service", "staff", "staff and customer service"],
            ["The service staff is very attentive and conscientious", "service staff", "conscientious"],
            ["they upgraded us to a beautiful suite without us even asking", "suite", "they"],
            ["10 points for the attitude and professionalism of the service staff", "service staff", "service"]
          ],
          [4000, 2000, 3026]
        ),
        caption("Bảng 4.9. Một số lỗi ATE tiêu biểu (dự đoán vs. gold)"),
        emptyLine(),

        heading3("4.5.2. Lỗi ở bước phân loại: nhãn thiểu số"),
        para("Ở mọi cấu hình, hai tổ hợp BRANDING_neutral và FACILITY_neutral đạt F1 = 0%, và SERVICE_negative thường rất thấp (0–50%). Đây là hệ quả trực tiếp của mất cân bằng dữ liệu: số mẫu của các tổ hợp này quá ít để mô hình học. Chính các nhãn này kéo Macro F1 (~48–57%) xuống thấp hơn nhiều so với Micro F1 (~71%). Đây là vấn đề dữ liệu, không phải lỗi kiến trúc – weighted loss và dữ liệu bổ trợ chỉ giảm nhẹ chứ không khắc phục triệt để.", { indent: true }),
        pageBreak(),

        // ═══════════════════════════════════════════════════════════════════
        // CHƯƠNG V - KẾT LUẬN
        // ═══════════════════════════════════════════════════════════════════
        heading1("CHƯƠNG V. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN"),

        heading2("5.1. Kết luận"),
        para("Khóa luận đã xây dựng và đánh giá hệ thống trích xuất bộ ba (aspect term, aspect category, sentiment) cho đánh giá khách sạn tiếng Việt, tích hợp hai phương pháp kỹ thuật trọng tâm là Local Context Focus (LCF) và Token Merging (ToMe). Các kết quả chính:", { indent: true }),
        numbered("Kiến trúc Joint vượt Pipeline khoảng 18 điểm Micro F1 (~71% so với ~53%). Oracle Joint Acc gần như không đổi giữa hai kiến trúc, xác nhận nút thắt nằm ở ATE – biểu hiện điển hình của lan truyền lỗi."),
        numbered("Mô hình tốt nhất: Joint · T5 · seq_resize đạt Micro F1 = 71.50% và Oracle Joint = 89.10%; Joint · BERT · lcf_seq_cdm_resize đạt Micro F1 = 71.18% với chi phí huấn luyện thấp hơn."),
        numbered("CDM tốt hơn CDW trên nhãn thiểu số: mặt nạ nhị phân CDM nâng Macro F1 thêm 6–9 điểm so với trọng số mềm CDW, trong khi Micro F1 gần như không đổi."),
        numbered("Gộp token không làm giảm chất lượng: các cấu hình ToMe đạt độ chính xác ngang hoặc nhỉnh hơn baseline, khẳng định tính khả thi của Token Merging trong ABSA."),
        emptyLine(),
        para("Trong chế độ resize, Token Merging không mang lại tăng tốc thực sự nhưng khẳng định được chất lượng biểu diễn. Đây là bước nền tảng cần thiết trước khi hiện thực hóa lợi ích tốc độ.", { indent: true }),

        heading2("5.2. Hạn chế"),
        bullet("Chưa đo tăng tốc thực sự: đóng góp của ToMe ở chế độ resize nằm ở khía cạnh chất lượng, chưa phải hiệu năng tính toán."),
        bullet("Nút thắt ATE: ATE F1 Pipeline chỉ ~60% do lỗi biên cụm từ và chọn nhầm khía cạnh."),
        bullet("Mất cân bằng dữ liệu: BRANDING_neutral, FACILITY_neutral đạt F1 = 0%; weighted loss và supplement chỉ giảm nhẹ vấn đề."),
        bullet("Quy mô đánh giá: bộ dữ liệu tương đối nhỏ (test 312 mẫu) và thuộc một miền, khả năng khái quát chưa được kiểm chứng."),
        emptyLine(),

        heading2("5.3. Hướng phát triển"),
        heading3("5.3.1. Hiện thực hóa tăng tốc Token Merging"),
        bullet("Chế độ compact (resize=False): giữ nguyên chuỗi rút gọn L' để các lớp SA phía sau xử lý ít token hơn, đo trực tiếp độ trễ và thông lượng."),
        bullet("Pre-BERT ToMe (use_pre_tome): gộp token ở mức embedding trước 12 lớp Transformer – đòn bẩy tốc độ lớn nhất vì rút ngắn chính phần O(n²) của backbone. Mã nguồn đã hỗ trợ (_pre_bert_merge) nhưng chưa đưa vào cấu hình chính."),
        bullet("Báo cáo đánh đổi tốc độ–độ chính xác: latency, FLOPs, throughput so với Micro/Macro F1."),

        heading3("5.3.2. Cải thiện bước ATE"),
        bullet("Tinh chỉnh T5 với ràng buộc biên chặt hơn (constrained decoding) để đầu ra luôn là chuỗi con hợp lệ."),
        bullet("Thử kiến trúc trích xuất span thay cho sinh tự do để giảm lỗi biên cụm từ."),

        heading3("5.3.3. Xử lý mất cân bằng và mở rộng dữ liệu"),
        bullet("Bổ sung dữ liệu cho nhãn hiếm; thử focal loss, over-sampling có kiểm soát, hoặc sinh dữ liệu tăng cường bằng LLM."),
        bullet("Mở rộng đánh giá sang miền khác (nhà hàng, thương mại điện tử) để kiểm chứng khả năng khái quát."),
        bullet("Nghiên cứu chiến lược gộp token nhận biết ranh giới khía cạnh tốt hơn và kết hợp tín hiệu attention thực thay vì trọng số đồng nhất."),
        emptyLine(),
        pageBreak(),

        // ═══════════════════════════════════════════════════════════════════
        // TÀI LIỆU THAM KHẢO
        // ═══════════════════════════════════════════════════════════════════
        heading1("TÀI LIỆU THAM KHẢO"),
        para("[1] Bolya, D., Fu, C. Y., Dai, X., Zhang, P., & Hoffman, J. (2023). Token merging: Your ViT but faster. CVPR 2023.", { indent: true }),
        para("[2] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL 2019.", { indent: true }),
        para("[3] Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140), 1-67.", { indent: true }),
        para("[4] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. NeurIPS 2017.", { indent: true }),
        para("[5] Li, X., Bing, L., Li, P., Lam, W., & Yang, Z. (2018). Aspect Term Extraction with History Attention and Selective Transformation. IJCAI 2018.", { indent: true }),
        para("[6] Pontiki, M., Galanis, D., Papageorgiou, H., Androutsopoulos, I., Manandhar, S., AL-Smadi, M., ... & Eryigit, G. (2016). SemEval-2016 task 5: Aspect based sentiment analysis. ACL Anthology.", { indent: true }),
        para("[7] Zhang, W., Li, X., Deng, Y., Bing, L., & Lam, W. (2021). Towards generative aspect-based sentiment analysis. ACL-IJCNLP 2021.", { indent: true }),
        para("[8] Yang, H., Li, Q., & Zeng, B. (2021). PyABSA: A Modularized Framework for Reproducible Aspect-based Sentiment Analysis. arXiv preprint.", { indent: true }),
        para("[9] Zeng, B., Yang, H., Xu, R., Zhou, W., & Han, X. (2019). LCF: A Local Context Focus Mechanism for Aspect-Based Sentiment Classification. Applied Sciences.", { indent: true }),
        para("[10] Phan, M. H., & Ogunbona, P. O. (2020). Modelling context and syntactical features for aspect-based sentiment analysis. ACL 2020.", { indent: true }),
        para("[11] Xu, H., Liu, B., Shu, L., & Philip, S. Y. (2020). BERT post-training for review reading comprehension and aspect-based sentiment analysis. NAACL 2019.", { indent: true }),
        para("[12] Sun, C., Huang, L., & Qiu, X. (2019). Utilizing BERT for aspect-based sentiment analysis via constructing auxiliary sentence. NAACL 2019.", { indent: true }),
        para("[13] Chen, Z., Qian, T. (2020). Relation-Aware Collaborative Learning for Unified Aspect-Based Sentiment Analysis. ACL 2020.", { indent: true }),
        emptyLine(),
        pageBreak(),

        // ═══════════════════════════════════════════════════════════════════
        // PHỤ LỤC
        // ═══════════════════════════════════════════════════════════════════
        heading1("PHỤ LỤC"),

        heading2("Phụ lục A. Cấu trúc thư mục dự án"),
        code("thesis_apc_baseline/"),
        code("├── src/                     # Module ATE – T5"),
        code("│   ├── train.py             # CLI huấn luyện T5 ATE"),
        code("│   ├── model.py             # T5AspectExtractor"),
        code("│   ├── trainer.py           # ATETrainer: AMP, early stopping"),
        code("│   ├── inference.py         # predict_aspects()"),
        code("│   ├── metrics.py           # Exact-match P/R/F1"),
        code("│   └── normalization.py     # n-gram Levenshtein normalization"),
        code("├── models/"),
        code("│   └── fast_lcf_bert_multitask.py  # BERT đa nhiệm: LCF + ToMe"),
        code("├── gas/                     # GAS – Generative Aspect Sentiment"),
        code("│   ├── model.py             # GasT5Model"),
        code("│   ├── train_gas.py, trainer.py, dataset.py"),
        code("│   ├── metrics.py           # 4-label ABSA metrics"),
        code("│   └── evaluate_joint.py"),
        code("├── experiments/             # Scripts huấn luyện và đánh giá"),
        code("│   ├── run_joint_experiments.py"),
        code("│   ├── eval_joint_triplet.py"),
        code("│   └── run_ate_inference.py"),
        code("├── token_merging/           # ToMe module"),
        code("│   └── tome_1d.py           # bipartite/sequential_local/attention_weighted"),
        code("├── dataset/                 # Dữ liệu .apc"),
        code("│   ├── train.apc / dev.apc / test.apc"),
        code("│   └── supplement/          # negative.tsv, neutral.tsv"),
        code("├── server/app.py            # FastAPI backend"),
        code("├── frontend/                # React + Vite"),
        code("├── pipeline_inference.py    # PipelineInference: ATE → APC"),
        code("└── requirements.txt"),
        emptyLine(),

        heading2("Phụ lục B. Hướng dẫn cài đặt và chạy"),
        heading3("B.1. Cài đặt môi trường"),
        code("pip install -r requirements.txt"),
        code("python -m spacy download en_core_web_sm"),
        emptyLine(),

        heading3("B.2. Huấn luyện ATE (T5)"),
        code("python src/train.py \\"),
        code("  --data-dir dataset \\"),
        code("  --output-dir checkpoints/gas_t5_ate \\"),
        code("  --model-name t5-base --epochs 20"),
        emptyLine(),

        heading3("B.3. Huấn luyện APC đa nhiệm (12 cấu hình)"),
        code("python experiments/run_joint_experiments.py"),
        emptyLine(),

        heading3("B.4. Đánh giá joint triplet"),
        code("python experiments/run_ate_inference.py"),
        code("python experiments/eval_joint_triplet.py --model-type bert"),
        code("python experiments/eval_joint_triplet.py --model-type t5"),
        emptyLine(),

        heading3("B.5. Inference pipeline"),
        code("python pipeline_inference.py \\"),
        code("  --ate-checkpoint checkpoints/gas_t5_ate/best \\"),
        code("  --apc-checkpoint-dir runs_joint/lcf_seq_cdm_resize \\"),
        code("  --sentence \"The room was beautiful but the staff was rude.\""),
        emptyLine(),

        heading3("B.6. Web Interface"),
        code("# Backend"),
        code("ATE_CHECKPOINT=checkpoints/gas_t5_ate/best \\"),
        code("APC_CHECKPOINT_DIR=runs_joint/lcf_seq_cdm_resize \\"),
        code("uvicorn server.app:app --host 0.0.0.0 --port 5000"),
        code(""),
        code("# Frontend"),
        code("cd frontend && npm install && npm run dev"),
        emptyLine(),

        heading2("Phụ lục C. Ví dụ đầu ra hệ thống"),
        heading3("C.1. Ví dụ inference pipeline"),
        code("Input: \"The room was spotless, but the elevator broke down frequently. Staff were incredibly helpful.\""),
        code("Output:"),
        code("["),
        code("  {\"aspect\": \"room\",     \"sentiment\": \"positive\", \"category\": \"FACILITY\"},"),
        code("  {\"aspect\": \"elevator\", \"sentiment\": \"negative\", \"category\": \"FACILITY\"},"),
        code("  {\"aspect\": \"staff\",    \"sentiment\": \"positive\", \"category\": \"SERVICE\"}"),
        code("]"),
        emptyLine(),

        heading3("C.2. Ví dụ GAS T5 inference"),
        code("Input:  \"The food was great\""),
        code("Output: [(food, AMENITY, positive)]"),
        emptyLine(),

        heading2("Phụ lục D. Ví dụ minh họa các chiến lược Token Merging"),
        heading3("D.1. Ví dụ Bipartite Merging"),
        para("Với câu \"I found the food delicious but service was terrible\" và khía cạnh food (vị trí 4):", { indent: true }),
        para("Pool nội bộ (trừ [CLS]=0, [SEP]=12, food=4): {1,2,3,5,6,7,8,9,10,11}. Chia xen kẽ: A={1,3,6,8,10}, B={2,5,7,9,11}.", { indent: true }),
        para("Token A=1 (\"i\") chọn B=7 (\"service\") với cosine ≈ 0.95 → gộp, B=7 loại. Token A=3 (\"the\") chọn B=2 (\"found\") → gộp. Kết quả: giảm từ 13 xuống 11 token sau 1 vòng.", { indent: true }),

        heading3("D.2. Ví dụ Sequential Local Merging"),
        para("Ranh giới LCF: vị trí 3 (\"the\") có lcf=0 và vị trí 4 (\"food\") có lcf=1 → mid_sep tại vị trí 3 (bảo vệ, không bị gộp).", { indent: true }),
        para("Khi quét tới vị trí 5 (\"delicious\"): sim với food (trái) ≈ 0.995 > sim với but (phải) ≈ 0.62 → \"delicious\" gập vào food. lcf[4] = max(1,0) = 1 – tín hiệu khía cạnh giữ nguyên. Hiệu ứng cascade: chuỗi rút mạnh xuống ~5 token sau 1 lượt.", { indent: true }),
        emptyLine()
      ]
    }
  ]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/mnt/user-data/outputs/luan_van_absa_token_merging.docx", buffer);
  console.log("Done: luan_van_absa_token_merging.docx");
}).catch(e => console.error(e));
