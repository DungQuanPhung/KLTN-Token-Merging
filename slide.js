const pptxgen = require("pptxgenjs");
const { buildIcons } = require("./icons.js");

// ---------- Palette (technical NLP / token-merging theme) ----------
const NAVY   = "0E2A47";  // deep navy - dark slide backgrounds
const NAVY2  = "16375C";  // lighter navy panel
const TEAL   = "12768A";  // primary teal
const TEAL_D = "0C5666";  // darker teal
const MINT   = "23B5A6";  // mint accent
const AMBER  = "E8A13A";  // amber highlight (key stats)
const CORAL  = "D96A5B";  // negative / contrast
const INK    = "1A2B3C";  // body text dark
const MUTE   = "5B6B7B";  // muted text
const LIGHT  = "F4F7F9";  // light panel bg
const LIGHT2 = "E9EFF3";  // light panel bg 2
const LINEC  = "D3DDE4";  // hairlines
const WHITE  = "FFFFFF";

// layout (WIDE 13.33 x 7.5)
const PW = 13.33, PH = 7.5, MX = 0.62;

const FS = "Calibri";              // safe body
const FH = "Cambria";              // safe serif for headers

const sh = () => ({ type: "outer", color: "0E2A47", blur: 9, offset: 3, angle: 90, opacity: 0.16 });
const shSoft = () => ({ type: "outer", color: "0E2A47", blur: 7, offset: 2, angle: 90, opacity: 0.10 });

let pres = new pptxgen();
pres.defineLayout({ name: "W", width: PW, height: PH });
pres.layout = "W";
pres.author = "Phùng Dũng Quân · Nguyễn Hồ Tuyên";
pres.title = "Attention-based Token Merging for ABSA (LCTA)";

let ICON;

// ---------- reusable helpers ----------

// content-slide header: eyebrow + title + slide number, icon chip
function header(slide, { kicker, title, icon, num }) {
  slide.background = { color: WHITE };
  // icon chip
  if (icon) {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: MX, y: 0.42, w: 0.62, h: 0.62, rectRadius: 0.12,
      fill: { color: TEAL }, line: { type: "none" }, shadow: shSoft(),
    });
    slide.addImage({ data: icon, x: MX + 0.14, y: 0.56, w: 0.34, h: 0.34 });
  }
  const tx = icon ? MX + 0.86 : MX;
  slide.addText(kicker.toUpperCase(), {
    x: tx, y: 0.4, w: 9.5, h: 0.28, margin: 0,
    fontFace: FS, fontSize: 11.5, bold: true, color: TEAL, charSpacing: 2, align: "left", valign: "bottom",
  });
  slide.addText(title, {
    x: tx, y: 0.66, w: 10.3, h: 0.56, margin: 0,
    fontFace: FH, fontSize: 25, bold: true, color: INK, align: "left", valign: "middle",
  });
  // slide number chip (top right)
  slide.addText(String(num).padStart(2, "0"), {
    x: PW - 1.15, y: 0.42, w: 0.6, h: 0.5, margin: 0,
    fontFace: FS, fontSize: 15, bold: true, color: LINEC, align: "right", valign: "middle",
  });
  // thin baseline under header using whitespace (no accent bar) -> use faint full separator only via light rule
  slide.addShape(pres.shapes.LINE, {
    x: MX, y: 1.34, w: PW - 2 * MX, h: 0,
    line: { color: LINEC, width: 1 },
  });
}

// token-pill motif strip (row of small rounded pills, some "merged")
function tokenStrip(slide, x, y, opts = {}) {
  const w = opts.w || 0.42, h = opts.h || 0.26, gap = opts.gap || 0.12;
  const tokens = opts.tokens || [TEAL, MINT, TEAL, LINEC, TEAL_D, MINT];
  let cx = x;
  tokens.forEach((col, i) => {
    const ww = (opts.merged && opts.merged.includes(i)) ? w * 1.7 : w;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: cx, y, w: ww, h, rectRadius: 0.06,
      fill: { color: col }, line: { type: "none" },
    });
    cx += ww + gap;
  });
}

function card(slide, x, y, w, h, fill = WHITE) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.09,
    fill: { color: fill }, line: { color: LINEC, width: 1 }, shadow: shSoft(),
  });
}

// numbered pipeline node
function pnode(slide, x, y, w, h, title, sub, color = TEAL) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.1,
    fill: { color: WHITE }, line: { color: color, width: 1.5 }, shadow: shSoft(),
  });
  slide.addText(title, {
    x: x + 0.08, y: y + 0.12, w: w - 0.16, h: 0.4, margin: 0,
    fontFace: FS, fontSize: 12.5, bold: true, color: color, align: "center", valign: "middle",
  });
  slide.addText(sub, {
    x: x + 0.08, y: y + 0.5, w: w - 0.16, h: h - 0.6, margin: 0,
    fontFace: FS, fontSize: 9.5, color: MUTE, align: "center", valign: "top",
  });
}

function arrow(slide, x, y, w = 0.34) {
  slide.addShape(pres.shapes.LINE, {
    x, y, w, h: 0,
    line: { color: TEAL, width: 2, endArrowType: "triangle" },
  });
}

// ==================================================================
async function main() {
  ICON = await buildIcons(TEAL.replace("#", ""));

  // ---------------- SLIDE 1 : TITLE ----------------
  {
    const s = pres.addSlide();
    s.background = { color: NAVY };
    // subtle panel band on right for depth (not an edge stripe: it's a large offset rounded panel)
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 8.9, y: -1.2, w: 6.2, h: 10, rectRadius: 0.3,
      fill: { color: NAVY2 }, line: { type: "none" },
    });
    // token merging motif top-right inside panel
    tokenStrip(s, 9.5, 1.15, { tokens: [TEAL, MINT, TEAL, TEAL_D, MINT, TEAL], w: 0.5, h: 0.3, gap: 0.14 });
    tokenStrip(s, 9.5, 1.75, { tokens: [TEAL, MINT, TEAL_D, MINT], w: 0.5, h: 0.3, gap: 0.14, merged: [0, 2] });
    tokenStrip(s, 9.5, 2.35, { tokens: [TEAL, MINT, TEAL_D], w: 0.5, h: 0.3, gap: 0.14, merged: [0, 1] });
    s.addImage({ data: ICON.layersW, x: 11.9, y: 3.35, w: 1.0, h: 1.0 });
    s.addText("Token Merging", { x: 9.3, y: 4.5, w: 3.6, h: 0.3, margin: 0, fontFace: FS, fontSize: 11, italic: true, color: MINT, align: "center" });

    s.addText("ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH  ·  TRƯỜNG ĐH KHOA HỌC TỰ NHIÊN", {
      x: MX, y: 0.7, w: 8.2, h: 0.3, margin: 0, fontFace: FS, fontSize: 11.5, bold: true, color: MINT, charSpacing: 1,
    });
    s.addText("KHÓA LUẬN TỐT NGHIỆP CỬ NHÂN  ·  NGÀNH KHOA HỌC DỮ LIỆU", {
      x: MX, y: 1.02, w: 8.2, h: 0.3, margin: 0, fontFace: FS, fontSize: 10.5, color: "9DB4C7", charSpacing: 1,
    });

    s.addText("Attention-based Token Merging", {
      x: MX, y: 1.95, w: 8.3, h: 0.72, margin: 0, fontFace: FH, fontSize: 40, bold: true, color: WHITE,
    });
    s.addText("for Aspect-Based Sentiment Analysis", {
      x: MX, y: 2.68, w: 8.3, h: 0.6, margin: 0, fontFace: FH, fontSize: 30, bold: true, color: MINT,
    });
    s.addText([
      { text: "Phương pháp ", options: {} },
      { text: "LCTA", options: { bold: true, color: WHITE } },
      { text: " — kết hợp Token Merging với cơ chế Local Context Focus (LCF) nhằm giảm dư thừa biểu diễn mà vẫn bảo toàn ngữ cảnh khía cạnh.", options: {} },
    ], {
      x: MX, y: 3.5, w: 7.9, h: 0.9, margin: 0, fontFace: FS, fontSize: 14.5, color: "C9D6E2", lineSpacingMultiple: 1.1,
    });

    // authors / advisors row
    s.addShape(pres.shapes.LINE, { x: MX, y: 4.75, w: 7.7, h: 0, line: { color: "2E4A66", width: 1 } });
    s.addText([
      { text: "NHÓM TÁC GIẢ", options: { bold: true, color: MINT, breakLine: true, fontSize: 10, charSpacing: 1 } },
      { text: "Phùng Dũng Quân   ·   Nguyễn Hồ Tuyên", options: { color: WHITE, fontSize: 14, bold: true } },
    ], { x: MX, y: 4.95, w: 4.2, h: 0.9, margin: 0, fontFace: FS, lineSpacingMultiple: 1.2 });
    s.addText([
      { text: "GIẢNG VIÊN HƯỚNG DẪN", options: { bold: true, color: MINT, breakLine: true, fontSize: 10, charSpacing: 1 } },
      { text: "ThS. Đoàn Thị Trâm", options: { color: WHITE, fontSize: 13, breakLine: true } },
      { text: "ThS. Lê Thị Tuyết Nhung", options: { color: WHITE, fontSize: 13 } },
    ], { x: 4.9, y: 4.95, w: 3.6, h: 1.0, margin: 0, fontFace: FS, lineSpacingMultiple: 1.15 });

    s.addText("Khoa Toán – Tin học  ·  TP. Hồ Chí Minh, 2026", {
      x: MX, y: 6.75, w: 8, h: 0.3, margin: 0, fontFace: FS, fontSize: 11, italic: true, color: "7E97AC",
    });
    s.addNotes("Slide tiêu đề. Đề tài: Attention-based Token Merging for ABSA — phương pháp đề xuất tên LCTA (Local Context Token-Merging Attention Network).");
  }

  // ---------------- SLIDE 2 : ABSA là gì ----------------
  {
    const s = pres.addSlide();
    header(s, { kicker: "Đặt vấn đề", title: "Phân tích cảm xúc theo khía cạnh (ABSA) là gì?", icon: ICON.comment, num: 2 });

    s.addText([
      { text: "ABSA", options: { bold: true, color: TEAL } },
      { text: " xác định cảm xúc gắn với ", options: {} },
      { text: "từng khía cạnh cụ thể", options: { bold: true, color: INK } },
      { text: " trong văn bản, thay vì gán một nhãn cảm xúc duy nhất cho cả câu như phương pháp truyền thống.", options: {} },
    ], { x: MX, y: 1.6, w: 6.0, h: 1.15, margin: 0, fontFace: FS, fontSize: 15, color: INK, lineSpacingMultiple: 1.2, valign: "top" });

    // example card
    card(s, MX, 2.95, 6.0, 3.7, LIGHT);
    s.addText("VÍ DỤ MINH HỌA", { x: MX + 0.3, y: 3.15, w: 5.4, h: 0.3, margin: 0, fontFace: FS, fontSize: 11, bold: true, color: MUTE, charSpacing: 1.5 });
    s.addText([
      { text: "“The room was ", options: { color: INK } },
      { text: "beautiful", options: { bold: true, color: MINT } },
      { text: " but the staff was ", options: { color: INK } },
      { text: "rude", options: { bold: true, color: CORAL } },
      { text: "”", options: { color: INK } },
    ], { x: MX + 0.3, y: 3.45, w: 5.4, h: 0.55, margin: 0, fontFace: FH, fontSize: 18, italic: true, valign: "middle" });

    // two mapping rows
    const mkRow = (yy, aspect, cat, senti, col) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.3, y: yy, w: 5.4, h: 0.72, rectRadius: 0.08, fill: { color: WHITE }, line: { color: LINEC, width: 1 } });
      s.addText(aspect, { x: MX + 0.45, y: yy, w: 1.55, h: 0.72, margin: 0, fontFace: FS, fontSize: 14, bold: true, color: INK, valign: "middle" });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 2.1, y: yy + 0.19, w: 1.55, h: 0.34, rectRadius: 0.17, fill: { color: TEAL }, line: { type: "none" } });
      s.addText(cat, { x: MX + 2.1, y: yy + 0.19, w: 1.55, h: 0.34, margin: 0, fontFace: FS, fontSize: 10.5, bold: true, color: WHITE, align: "center", valign: "middle" });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 3.85, y: yy + 0.19, w: 1.55, h: 0.34, rectRadius: 0.17, fill: { color: col }, line: { type: "none" } });
      s.addText(senti, { x: MX + 3.85, y: yy + 0.19, w: 1.55, h: 0.34, margin: 0, fontFace: FS, fontSize: 10.5, bold: true, color: WHITE, align: "center", valign: "middle" });
    };
    mkRow(4.25, "room", "FACILITY", "POSITIVE", MINT);
    mkRow(5.15, "staff", "SERVICE", "NEGATIVE", CORAL);
    s.addText("Hai cảm xúc trái ngược cùng tồn tại trong một câu — điều mà cảm xúc mức câu không nắm bắt được.", {
      x: MX + 0.3, y: 5.95, w: 5.4, h: 0.55, margin: 0, fontFace: FS, fontSize: 11, italic: true, color: MUTE, valign: "top",
    });

    // right column: why it matters
    card(s, 7.0, 1.6, PW - 7.0 - MX, 5.05, WHITE);
    s.addImage({ data: ICON.bullseye, x: 7.35, y: 1.95, w: 0.5, h: 0.5 });
    s.addText("Vì sao ABSA quan trọng?", { x: 8.0, y: 1.95, w: 4.6, h: 0.5, margin: 0, fontFace: FH, fontSize: 17, bold: true, color: INK, valign: "middle" });
    const bullets = [
      ["Đánh giá bùng nổ", "Nền tảng đặt phòng & mạng xã hội du lịch tạo ra lượng review khổng lồ, mỗi câu đề cập nhiều khía cạnh với sắc thái khác nhau."],
      ["Ứng dụng thực tế", "Giúp doanh nghiệp xác định chính xác điểm mạnh – điểm yếu của từng dịch vụ để điều chỉnh phù hợp."],
      ["Ba bài toán nền tảng", "ATE (trích cụm khía cạnh) → ACSD (nhóm khía cạnh + cảm xúc) → TASD (bộ ba đầy đủ)."],
    ];
    let yy = 2.7;
    bullets.forEach(([t, d]) => {
      s.addImage({ data: ICON.check, x: 7.35, y: yy + 0.03, w: 0.28, h: 0.28 });
      s.addText([
        { text: t + ".  ", options: { bold: true, color: TEAL } },
        { text: d, options: { color: INK } },
      ], { x: 7.78, y: yy - 0.05, w: 4.85, h: 1.15, margin: 0, fontFace: FS, fontSize: 12.5, lineSpacingMultiple: 1.12, valign: "top" });
      yy += 1.28;
    });
    s.addNotes("ABSA làm rõ cảm xúc theo từng khía cạnh. Ví dụ kinh điển: room đẹp (tích cực) nhưng staff thô lỗ (tiêu cực).");
  }

  // ---------------- SLIDE 3 : Động cơ & khoảng trống ----------------
  {
    const s = pres.addSlide();
    header(s, { kicker: "Động cơ nghiên cứu", title: "Chi phí Transformer & khoảng trống của Token Merging", icon: ICON.warning, num: 3 });

    // Problem 1: O(L^2)
    card(s, MX, 1.6, 3.85, 3.05);
    s.addImage({ data: ICON.compress, x: MX + 0.28, y: 1.85, w: 0.5, h: 0.5 });
    s.addText("Chi phí self-attention", { x: MX + 0.28, y: 2.45, w: 3.3, h: 0.35, margin: 0, fontFace: FS, fontSize: 14, bold: true, color: INK });
    s.addText([{ text: "O(L", options: {} }, { text: "2", options: { superscript: true } }, { text: ")", options: {} }], {
      x: MX + 0.28, y: 2.8, w: 3.3, h: 0.75, margin: 0, fontFace: FH, fontSize: 42, bold: true, color: TEAL,
    });
    s.addText("Chi phí tăng theo bình phương độ dài chuỗi L. Câu dài chứa nhiều token dư thừa (từ dừng, dấu câu) → suy luận chậm, tốn tài nguyên.", {
      x: MX + 0.28, y: 3.55, w: 3.3, h: 0.95, margin: 0, fontFace: FS, fontSize: 11.5, color: MUTE, lineSpacingMultiple: 1.1, valign: "top",
    });

    // Problem 2: ToMe risk
    card(s, MX + 4.05, 1.6, 3.85, 3.05);
    s.addImage({ data: ICON.warning, x: MX + 4.33, y: 1.85, w: 0.5, h: 0.5 });
    s.addText("Rủi ro khi áp dụng ToMe", { x: MX + 4.33, y: 2.45, w: 3.3, h: 0.35, margin: 0, fontFace: FS, fontSize: 14, bold: true, color: INK });
    tokenStrip(s, MX + 4.33, 2.95, { tokens: [TEAL, CORAL, TEAL, LINEC, TEAL_D], w: 0.42, h: 0.28, gap: 0.1, merged: [1] });
    s.addText("Token thuộc cụm khía cạnh có thể bị hợp nhất chỉ dựa trên độ tương đồng biểu diễn, làm mất thông tin ngữ nghĩa quyết định cảm xúc.", {
      x: MX + 4.33, y: 3.45, w: 3.3, h: 1.05, margin: 0, fontFace: FS, fontSize: 11.5, color: MUTE, lineSpacingMultiple: 1.1, valign: "top",
    });

    // gap panel (dark)
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 8.1, y: 1.6, w: PW - (MX + 8.1) - MX, h: 3.05, rectRadius: 0.1, fill: { color: NAVY }, line: { type: "none" }, shadow: sh() });
    s.addText("KHOẢNG TRỐNG NGHIÊN CỨU", { x: MX + 8.4, y: 1.9, w: 3.5, h: 0.3, margin: 0, fontFace: FS, fontSize: 11, bold: true, color: MINT, charSpacing: 1.5 });
    s.addText([
      { text: "Chưa xét đầy đủ tương tác giữa ToMe và LCF trong mô hình ABSA đa nhiệm.", options: { breakLine: true, bullet: { indent: 14 } } },
      { text: "Thiếu khảo sát hệ thống nhiều chiến lược hợp nhất token khác nhau.", options: { bullet: { indent: 14 } } },
    ], { x: MX + 8.4, y: 2.3, w: 3.55, h: 2.2, margin: 0, fontFace: FS, fontSize: 13, color: "D5E1EC", lineSpacingMultiple: 1.2, paraSpaceAfter: 10, valign: "top" });

    // conclusion strip -> LCTA
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 4.95, w: PW - 2 * MX, h: 1.55, rectRadius: 0.1, fill: { color: LIGHT }, line: { color: TEAL, width: 1.5 } });
    s.addImage({ data: ICON.rocket, x: MX + 0.35, y: 5.35, w: 0.75, h: 0.75 });
    s.addText([
      { text: "Đề xuất phương pháp LCTA", options: { bold: true, color: TEAL, fontSize: 18, breakLine: true } },
      { text: "Local Context Token-Merging Attention Network — xây dựng & đánh giá thực nghiệm các chiến lược hợp nhất token mới kết hợp LCF, tìm cân bằng giữa hiệu quả tính toán và độ chính xác trên dữ liệu đánh giá khách sạn thực tế.", options: { color: INK, fontSize: 13 } },
    ], { x: MX + 1.35, y: 5.15, w: PW - 2 * MX - 1.7, h: 1.2, margin: 0, fontFace: FS, lineSpacingMultiple: 1.12, valign: "middle" });
    s.addNotes("Hai vấn đề: chi phí O(L^2) và rủi ro mất thông tin aspect khi merge. Khoảng trống: chưa nghiên cứu ToMe×LCF và nhiều chiến lược. → LCTA.");
  }

  // ---------------- SLIDE 4 : Mục tiêu ----------------
  {
    const s = pres.addSlide();
    header(s, { kicker: "Mục tiêu nghiên cứu", title: "Bảy mục tiêu cụ thể của khóa luận", icon: ICON.bullseye, num: 4 });

    const goals = [
      [ICON.sitemap, "Hệ thống ABSA hai giai đoạn", "Giải TASD qua ATE (sinh có điều kiện dựa trên T5) và ACSD (phân loại đồng thời)."],
      [ICON.cogs, "Bộ phân loại ACSD đa nhiệm", "Dựa trên LCF, dự đoán 3 lớp cảm xúc và 6 nhóm khía cạnh cùng lúc."],
      [ICON.layers, "Tích hợp mô-đun ToMe", "Bật/tắt độc lập với LCF; hỗ trợ hai chế độ đầu ra resize và compact."],
      [ICON.route, "Hai chiến lược ToMe mới", "Đề xuất SLM và SCM bên cạnh chiến lược gốc BiToMe."],
      [ICON.flask, "Bộ thực nghiệm có kiểm soát", "12 cấu hình kết hợp {LCF × ToMe × chiến lược} để đo Micro-F1 & Macro-F1."],
      [ICON.database, "Xây dựng & xử lý dữ liệu", "Bộ dữ liệu ABSA khách sạn (tiếng Anh) kèm tăng cường dữ liệu giảm mất cân bằng."],
    ];
    const cardW = (PW - 2 * MX - 2 * 0.3) / 3, cardH = 1.72;
    goals.forEach((g, i) => {
      const col = i % 3, row = Math.floor(i / 3);
      const x = MX + col * (cardW + 0.3);
      const y = 1.6 + row * (cardH + 0.3);
      card(s, x, y, cardW, cardH);
      s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.25, w: 0.6, h: 0.6, fill: { color: LIGHT2 }, line: { type: "none" } });
      s.addImage({ data: g[0], x: x + 0.38, y: y + 0.38, w: 0.34, h: 0.34 });
      s.addText(String(i + 1), { x: x + cardW - 0.7, y: y + 0.15, w: 0.55, h: 0.5, margin: 0, fontFace: FH, fontSize: 26, bold: true, color: LIGHT2, align: "right" });
      s.addText(g[1], { x: x + 0.25, y: y + 0.92, w: cardW - 0.5, h: 0.35, margin: 0, fontFace: FS, fontSize: 12.5, bold: true, color: TEAL });
      s.addText(g[2], { x: x + 0.25, y: y + 1.24, w: cardW - 0.5, h: 0.44, margin: 0, fontFace: FS, fontSize: 10.3, color: MUTE, lineSpacingMultiple: 1.05, valign: "top" });
    });
    // 7th goal ribbon
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 1.6 + 2 * (cardH + 0.3), w: PW - 2 * MX, h: 0.62, rectRadius: 0.08, fill: { color: NAVY }, line: { type: "none" }, shadow: shSoft() });
    s.addText([
      { text: "7   ", options: { bold: true, color: MINT, fontSize: 20 } },
      { text: "Quy trình suy luận đầu–cuối hoàn chỉnh:", options: { bold: true, color: WHITE } },
      { text: "  câu dài được tách thành đơn vị ý kiến bằng LLM Qwen3-8B → trích aspect bằng T5 → phân loại ACSD → bộ ba (aspect term, category, sentiment).", options: { color: "D5E1EC" } },
    ], { x: MX + 0.3, y: 1.6 + 2 * (cardH + 0.3), w: PW - 2 * MX - 0.6, h: 0.62, margin: 0, fontFace: FS, fontSize: 12, valign: "middle" });
    s.addNotes("7 mục tiêu: pipeline 2 giai đoạn, classifier ACSD đa nhiệm, mô-đun ToMe, 2 chiến lược mới SLM/SCM, 12 cấu hình, dữ liệu + augmentation, pipeline end-to-end với LLM.");
  }

  // ---------------- SLIDE 5 : Câu hỏi nghiên cứu ----------------
  {
    const s = pres.addSlide();
    header(s, { kicker: "Câu hỏi nghiên cứu", title: "Bốn câu hỏi được kiểm chứng định lượng", icon: ICON.question, num: 5 });

    const rqs = [
      ["RQ1", "ToMe có duy trì / cải thiện hiệu năng ACSD so với Baseline không, đồng thời giảm dư thừa trong biểu diễn ngữ cảnh?", TEAL],
      ["RQ2", "Chiến lược SCM (cosine toàn chuỗi) có ưu việt hơn BiToMe (ghép cặp hai phía) trong câu ngắn của ABSA không? Ba chiến lược khác nhau ra sao?", MINT],
      ["RQ3", "Sự kết hợp LCF (CDM / CDW) và ToMe tác động thế nào đến khả năng bảo toàn thông tin cục bộ liên quan đến khía cạnh?", TEAL_D],
      ["RQ4", "Bước tiền xử lý tách câu bằng LLM ảnh hưởng ra sao đến độ bao phủ (Recall) của hệ thống ở bài toán TASD?", AMBER],
    ];
    const cw = (PW - 2 * MX - 0.35) / 2, chh = 2.32;
    rqs.forEach((r, i) => {
      const col = i % 2, row = Math.floor(i / 2);
      const x = MX + col * (cw + 0.35);
      const y = 1.65 + row * (chh + 0.3);
      card(s, x, y, cw, chh);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.3, y: y + 0.32, w: 1.15, h: 0.6, rectRadius: 0.1, fill: { color: r[2] }, line: { type: "none" } });
      s.addText(r[0], { x: x + 0.3, y: y + 0.32, w: 1.15, h: 0.6, margin: 0, fontFace: FH, fontSize: 21, bold: true, color: WHITE, align: "center", valign: "middle" });
      s.addText(r[1], { x: x + 1.65, y: y + 0.3, w: cw - 1.95, h: chh - 0.6, margin: 0, fontFace: FS, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.2, valign: "middle" });
    });
    s.addNotes("4 câu hỏi nghiên cứu RQ1-RQ4 — trả lời ở Chương 4.");
  }

  // ---------------- SLIDE 6 : 3 bài toán con ----------------
  {
    const s = pres.addSlide();
    header(s, { kicker: "Cơ sở lý thuyết", title: "Ba bài toán con của ABSA & mô hình nền", icon: ICON.sitemap, num: 6 });

    const tasks = [
      ["ATE", "Aspect Term Extraction", "Trích xuất các cụm từ khía cạnh xuất hiện trong câu.", "(room); (staff); (elevator)"],
      ["ACSD", "Aspect Category & Sentiment Detection", "Với mỗi cụm từ khía cạnh: đồng thời xác định nhóm khía cạnh + cực cảm xúc.", "room → FACILITY / positive"],
      ["TASD", "Target-Aspect-Sentiment Detection", "Bài toán hợp nhất: phát hiện trọn bộ ba cho mỗi khía cạnh.", "(room, FACILITY, positive)"],
    ];
    const cw = (PW - 2 * MX - 2 * 0.4) / 3, chh = 2.85;
    tasks.forEach((t, i) => {
      const x = MX + i * (cw + 0.4), y = 1.62;
      card(s, x, y, cw, chh);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x, y: y, w: cw, h: 0.72, rectRadius: 0.09, fill: { color: [TEAL, TEAL_D, NAVY][i] }, line: { type: "none" } });
      s.addText(t[0], { x: x, y: y, w: cw, h: 0.72, margin: 0, fontFace: FH, fontSize: 24, bold: true, color: WHITE, align: "center", valign: "middle" });
      s.addText(t[1], { x: x + 0.25, y: y + 0.9, w: cw - 0.5, h: 0.65, margin: 0, fontFace: FS, fontSize: 12.5, bold: true, color: INK, align: "center" });
      s.addText(t[2], { x: x + 0.25, y: y + 1.55, w: cw - 0.5, h: 0.72, margin: 0, fontFace: FS, fontSize: 11.5, color: MUTE, align: "center", lineSpacingMultiple: 1.1, valign: "top" });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.25, y: y + chh - 0.6, w: cw - 0.5, h: 0.4, rectRadius: 0.08, fill: { color: LIGHT }, line: { type: "none" } });
      s.addText(t[3], { x: x + 0.25, y: y + chh - 0.6, w: cw - 0.5, h: 0.4, margin: 0, fontFace: "Courier New", fontSize: 10, italic: true, color: TEAL, align: "center", valign: "middle" });
      if (i < 2) arrow(s, x + cw + 0.03, y + chh / 2, 0.34);
    });

    // backbones row
    s.addText("HAI BỘ MÃ HÓA NỀN (BACKBONE)", { x: MX, y: 4.75, w: 8, h: 0.3, margin: 0, fontFace: FS, fontSize: 11, bold: true, color: MUTE, charSpacing: 1.5 });
    const bb = [
      ["BERT", "bert-base-uncased", "Encoder 12 lớp · hidden 768 · ~110M tham số · backbone phổ biến nhất trong ABSA."],
      ["T5 Encoder", "t5-base", "Encoder T5 12 lớp · 768 · ~220M · relative positional encoding, phù hợp chuỗi độ dài thay đổi."],
    ];
    bb.forEach((b, i) => {
      const x = MX + i * ((PW - 2 * MX - 0.35) / 2 + 0.35), w = (PW - 2 * MX - 0.35) / 2;
      card(s, x, 5.1, w, 1.4, LIGHT);
      s.addImage({ data: ICON.brain, x: x + 0.3, y: 5.42, w: 0.55, h: 0.55 });
      s.addText([{ text: b[0], options: { bold: true, color: TEAL, fontSize: 16 } }, { text: "   " + b[1], options: { color: MUTE, fontSize: 11, italic: true } }], { x: x + 1.0, y: 5.38, w: w - 1.3, h: 0.4, margin: 0, fontFace: FS });
      s.addText(b[2], { x: x + 1.0, y: 5.78, w: w - 1.3, h: 0.6, margin: 0, fontFace: FS, fontSize: 11, color: INK, lineSpacingMultiple: 1.08, valign: "top" });
    });
    s.addNotes("ATE → ACSD → TASD. Hai backbone: BERT và T5 encoder.");
  }

  // ---------------- SLIDE 7 : LCF ----------------
  {
    const s = pres.addSlide();
    header(s, { kicker: "Cơ sở lý thuyết", title: "Local Context Focus (LCF): CDM & CDW", icon: ICON.crosshairs, num: 7 });

    s.addText([
      { text: "Ý tưởng: ", options: { bold: true, color: TEAL } },
      { text: "không phải token nào cũng đóng góp như nhau. Token gần khía cạnh chứa nhiều tín hiệu cảm xúc; token ở xa có thể thuộc khía cạnh khác và gây nhiễu.", options: { color: INK } },
    ], { x: MX, y: 1.55, w: PW - 2 * MX, h: 0.6, margin: 0, fontFace: FS, fontSize: 14, lineSpacingMultiple: 1.15, valign: "top" });

    // SRD box
    card(s, MX, 2.3, 4.1, 4.2, LIGHT);
    s.addText("SRD — Khoảng cách ngữ nghĩa tương đối", { x: MX + 0.3, y: 2.55, w: 3.5, h: 0.55, margin: 0, fontFace: FS, fontSize: 13.5, bold: true, color: INK });
    s.addText("Đo mức độ “gần” của token so với cụm khía cạnh:", { x: MX + 0.3, y: 3.1, w: 3.5, h: 0.5, margin: 0, fontFace: FS, fontSize: 11.5, color: MUTE });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.3, y: 3.6, w: 3.5, h: 0.6, rectRadius: 0.08, fill: { color: WHITE }, line: { color: LINEC, width: 1 } });
    s.addText("SRDᵢ = max(0, |i − c| − ⌊lₐ/2⌋)", { x: MX + 0.3, y: 3.6, w: 3.5, h: 0.6, margin: 0, fontFace: "Courier New", fontSize: 12.5, bold: true, color: TEAL, align: "center", valign: "middle" });
    s.addText([
      { text: "Token trong cụm khía cạnh: SRD = 0.", options: { bullet: { indent: 12 }, breakLine: true } },
      { text: "Càng ra xa biên, SRD tăng tuyến tính.", options: { bullet: { indent: 12 }, breakLine: true } },
      { text: "Ngưỡng bán kính α = 5 xác định vùng ngữ cảnh cục bộ (nhận trọng số tối đa = 1).", options: { bullet: { indent: 12 } } },
    ], { x: MX + 0.3, y: 4.35, w: 3.55, h: 2.0, margin: 0, fontFace: FS, fontSize: 12, color: INK, lineSpacingMultiple: 1.12, paraSpaceAfter: 8, valign: "top" });

    // CDM
    card(s, MX + 4.35, 2.3, 4.05, 4.2);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 4.35, y: 2.3, w: 4.05, h: 0.62, rectRadius: 0.09, fill: { color: TEAL }, line: { type: "none" } });
    s.addText("CDM · Context Dynamic Masking", { x: MX + 4.55, y: 2.3, w: 3.7, h: 0.62, margin: 0, fontFace: FS, fontSize: 13.5, bold: true, color: WHITE, valign: "middle" });
    s.addText("Mặt nạ nhị phân — cắt cứng.", { x: MX + 4.55, y: 3.05, w: 3.65, h: 0.35, margin: 0, fontFace: FS, fontSize: 12, italic: true, color: TEAL });
    s.addText("Che hoàn toàn (triệt tiêu) các token nằm ngoài vùng ngữ cảnh khía cạnh (SRD > α). Chỉ giữ lại vùng cục bộ.", { x: MX + 4.55, y: 3.45, w: 3.65, h: 0.95, margin: 0, fontFace: FS, fontSize: 12, color: INK, lineSpacingMultiple: 1.12, valign: "top" });
    // mask visual
    tokenStrip(s, MX + 4.55, 4.5, { tokens: [LINEC, LINEC, TEAL, TEAL, MINT, TEAL, LINEC, LINEC], w: 0.36, h: 0.3, gap: 0.08 });
    s.addText("token bị che  ·  vùng cục bộ giữ lại", { x: MX + 4.55, y: 4.95, w: 3.65, h: 0.3, margin: 0, fontFace: FS, fontSize: 9.5, italic: true, color: MUTE });
    s.addText("Phù hợp với các chiến lược chọn/gộp token theo độ quan trọng toàn cục (SCM, BiToMe).", { x: MX + 4.55, y: 5.4, w: 3.65, h: 0.9, margin: 0, fontFace: FS, fontSize: 11, color: MUTE, lineSpacingMultiple: 1.1, valign: "top" });

    // CDW
    card(s, MX + 8.65, 2.3, PW - (MX + 8.65) - MX, 4.2);
    const cdwW = PW - (MX + 8.65) - MX;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 8.65, y: 2.3, w: cdwW, h: 0.62, rectRadius: 0.09, fill: { color: TEAL_D }, line: { type: "none" } });
    s.addText("CDW · Context Dynamic Weighting", { x: MX + 8.85, y: 2.3, w: cdwW - 0.4, h: 0.62, margin: 0, fontFace: FS, fontSize: 13.5, bold: true, color: WHITE, valign: "middle" });
    s.addText("Trọng số giảm dần — làm mờ mềm.", { x: MX + 8.85, y: 3.05, w: cdwW - 0.4, h: 0.35, margin: 0, fontFace: FS, fontSize: 12, italic: true, color: TEAL_D });
    s.addText("Gán trọng số liên tục wᵢ ∈ [0,1] tỉ lệ nghịch với khoảng cách. Token xa chỉ bị làm mờ dần, không xóa hẳn → giữ một phần ngữ cảnh toàn cục.", { x: MX + 8.85, y: 3.45, w: cdwW - 0.4, h: 1.0, margin: 0, fontFace: FS, fontSize: 12, color: INK, lineSpacingMultiple: 1.12, valign: "top" });
    tokenStrip(s, MX + 8.85, 4.5, { tokens: ["9DC9C4", "5FBBB2", MINT, TEAL, MINT, "5FBBB2", "9DC9C4", "C9DDDA"], w: 0.36, h: 0.3, gap: 0.08 });
    s.addText("đóng góp mờ dần theo khoảng cách", { x: MX + 8.85, y: 4.95, w: cdwW - 0.4, h: 0.3, margin: 0, fontFace: FS, fontSize: 9.5, italic: true, color: MUTE });
    s.addText("Phù hợp hơn với hợp nhất tuần tự theo vùng cục bộ (SLM).", { x: MX + 8.85, y: 5.4, w: cdwW - 0.4, h: 0.9, margin: 0, fontFace: FS, fontSize: 11, color: MUTE, lineSpacingMultiple: 1.1, valign: "top" });
    s.addNotes("LCF: SRD đo khoảng cách tới aspect. CDM cắt cứng (binary mask), CDW làm mờ mềm (continuous weight).");
  }

  // ---------------- SLIDE 8 : Token Merging ----------------
  {
    const s = pres.addSlide();
    header(s, { kicker: "Cơ sở lý thuyết", title: "Token Merging (ToMe) & hai chế độ đầu ra", icon: ICON.layers, num: 8 });

    // concept left
    card(s, MX, 1.6, 6.0, 2.55, LIGHT);
    s.addText("Ý tưởng cốt lõi", { x: MX + 0.3, y: 1.82, w: 5.4, h: 0.35, margin: 0, fontFace: FS, fontSize: 14, bold: true, color: INK });
    s.addText("ToMe hợp nhất các token có biểu diễn tương đồng (đo bằng độ tương đồng cosine) thành một token đại diện, giảm dần số token mà không cần huấn luyện lại. Ban đầu đề xuất cho Vision Transformer.", {
      x: MX + 0.3, y: 2.2, w: 5.4, h: 0.95, margin: 0, fontFace: FS, fontSize: 12.5, color: INK, lineSpacingMultiple: 1.15, valign: "top",
    });
    // before/after token strips
    s.addText("Trước", { x: MX + 0.3, y: 3.25, w: 0.9, h: 0.3, margin: 0, fontFace: FS, fontSize: 10.5, bold: true, color: MUTE });
    tokenStrip(s, MX + 1.15, 3.28, { tokens: [TEAL, MINT, TEAL, MINT, TEAL_D, MINT, TEAL], w: 0.4, h: 0.26, gap: 0.08 });
    s.addText("Sau", { x: MX + 0.3, y: 3.68, w: 0.9, h: 0.3, margin: 0, fontFace: FS, fontSize: 10.5, bold: true, color: MUTE });
    tokenStrip(s, MX + 1.15, 3.7, { tokens: [TEAL, MINT, TEAL_D, MINT], w: 0.4, h: 0.26, gap: 0.08, merged: [0, 1] });

    // three strategies teaser
    card(s, MX, 4.35, 6.0, 2.2, WHITE);
    s.addText("Ba chiến lược khảo sát trong khóa luận", { x: MX + 0.3, y: 4.55, w: 5.4, h: 0.35, margin: 0, fontFace: FS, fontSize: 13.5, bold: true, color: TEAL });
    const strat = [["BiToMe", "Ghép cặp hai phía (gốc ToMe)"], ["SLM", "Hợp nhất cục bộ tuần tự"], ["SCM", "Hợp nhất tuần tự theo cosine toàn chuỗi"]];
    let sy = 4.98;
    strat.forEach(([a, b]) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.3, y: sy, w: 1.35, h: 0.42, rectRadius: 0.09, fill: { color: LIGHT2 }, line: { type: "none" } });
      s.addText(a, { x: MX + 0.3, y: sy, w: 1.35, h: 0.42, margin: 0, fontFace: FS, fontSize: 12, bold: true, color: TEAL_D, align: "center", valign: "middle" });
      s.addText(b, { x: MX + 1.8, y: sy, w: 3.9, h: 0.42, margin: 0, fontFace: FS, fontSize: 11.5, color: INK, valign: "middle" });
      sy += 0.5;
    });

    // resize vs compact comparison table (right)
    s.addText("HAI CHẾ ĐỘ ĐẦU RA SAU HỢP NHẤT", { x: 6.9, y: 1.6, w: 6, h: 0.3, margin: 0, fontFace: FS, fontSize: 11, bold: true, color: MUTE, charSpacing: 1.5 });
    const rows = [
      [{ text: "Thuộc tính", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "left" } },
       { text: "Nội suy (resize)", options: { bold: true, color: WHITE, fill: { color: TEAL } } },
       { text: "Nén (compact)", options: { bold: true, color: WHITE, fill: { color: TEAL_D } } }],
      ["Độ dài đầu ra", "L (giữ nguyên)", "L′ ≤ L"],
      ["Thay đổi tensor", "Không", "Có"],
      ["Tăng tốc suy luận thực sự", "Không", "Có"],
      ["Mục tiêu đánh giá", "Chất lượng biểu diễn", "Tốc độ + chất lượng"],
      [{ text: "Dùng trong TN chính", options: { bold: true } }, { text: "✓  (α, L=128)", options: { bold: true, color: TEAL } }, { text: "chỉ bổ sung", options: { color: MUTE } }],
    ];
    s.addTable(rows, {
      x: 6.9, y: 1.95, w: PW - 6.9 - MX, colW: [2.3, 1.95, 1.56],
      rowH: [0.44, 0.42, 0.42, 0.42, 0.42, 0.44],
      fontFace: FS, fontSize: 11.5, color: INK, valign: "middle", align: "center",
      border: { pt: 1, color: LINEC }, fill: { color: WHITE },
    });
    // note under table
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.9, y: 5.05, w: PW - 6.9 - MX, h: 1.5, rectRadius: 0.1, fill: { color: LIGHT }, line: { color: LINEC, width: 1 } });
    s.addImage({ data: ICON.balance, x: 7.15, y: 5.32, w: 0.5, h: 0.5 });
    s.addText([
      { text: "Vì sao TN chính chỉ dùng resize?  ", options: { bold: true, color: TEAL } },
      { text: "Giữ độ dài chuỗi cố định (L=128) để cô lập biến cần đánh giá — mọi khác biệt hiệu năng quy trực tiếp cho chiến lược hợp nhất token, không bị nhiễu bởi thay đổi độ dài.", options: { color: INK } },
    ], { x: 7.8, y: 5.2, w: PW - 6.9 - MX - 1.1, h: 1.2, margin: 0, fontFace: FS, fontSize: 11.5, lineSpacingMultiple: 1.1, valign: "middle" });
    s.addNotes("ToMe = gộp token tương đồng cosine. Resize nội suy về L gốc; compact giữ chuỗi ngắn. TN chính dùng resize để so sánh công bằng.");
  }

  // ---------------- SLIDE 9 : Kiến trúc tổng thể LCTA ----------------
  {
    const s = pres.addSlide();
    s.background = { color: NAVY };
    // header dark variant
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 0.42, w: 0.62, h: 0.62, rectRadius: 0.12, fill: { color: MINT }, line: { type: "none" } });
    s.addImage({ data: ICON.projectW || ICON.project, x: MX + 0.14, y: 0.56, w: 0.34, h: 0.34 });
    s.addText("PHƯƠNG PHÁP ĐỀ XUẤT", { x: MX + 0.86, y: 0.4, w: 9, h: 0.28, margin: 0, fontFace: FS, fontSize: 11.5, bold: true, color: MINT, charSpacing: 2, valign: "bottom" });
    s.addText("LCTA — Kiến trúc tổng thể pipeline hai giai đoạn", { x: MX + 0.86, y: 0.66, w: 11, h: 0.56, margin: 0, fontFace: FH, fontSize: 25, bold: true, color: WHITE, valign: "middle" });
    s.addText("09", { x: PW - 1.15, y: 0.42, w: 0.6, h: 0.5, margin: 0, fontFace: FS, fontSize: 15, bold: true, color: "3A5876", align: "right", valign: "middle" });

    // Big flow: Input -> Preprocess(UOS) -> Stage1 ATE -> Stage2 ACSD -> TASD
    const pnodeD = (x, y, w, h, tag, title, sub, accent) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: NAVY2 }, line: { color: accent, width: 1.5 } });
      s.addText(tag, { x: x + 0.12, y: y + 0.12, w: w - 0.24, h: 0.3, margin: 0, fontFace: FS, fontSize: 10, bold: true, color: accent, charSpacing: 1 });
      s.addText(title, { x: x + 0.12, y: y + 0.4, w: w - 0.24, h: 0.4, margin: 0, fontFace: FS, fontSize: 14, bold: true, color: WHITE });
      s.addText(sub, { x: x + 0.12, y: y + 0.82, w: w - 0.24, h: h - 0.9, margin: 0, fontFace: FS, fontSize: 10.5, color: "AEC2D4", lineSpacingMultiple: 1.1, valign: "top" });
    };
    const y0 = 1.75, hh = 2.15;
    pnodeD(MX, y0, 2.55, hh, "TIỀN XỬ LÝ", "Tách đơn vị ý kiến (UOS)", "Câu dài → nhiều UOS độc lập bằng LLM (Qwen3-8B), cô lập ngữ cảnh từng khía cạnh.", MINT);
    pnodeD(MX + 3.0, y0, 2.55, hh, "GIAI ĐOẠN 1 · ATE", "Trích cụm khía cạnh", "Mô hình sinh dựa trên T5 (GAS) → danh sách aspect; chuẩn hóa bằng Levenshtein.", TEAL);
    pnodeD(MX + 6.0, y0, 3.55, hh, "GIAI ĐOẠN 2 · ACSD", "Phân loại đa nhiệm", "Encoder → ToMe → Resize/Compact → LCF → Feature Fusion → Self-Attention → 2 đầu ra.", TEAL);
    pnodeD(MX + 10.0, y0, PW - (MX + 10.0) - MX, hh, "ĐẦU RA", "Bộ ba TASD", "Ghép kết quả 2 giai đoạn → tập bộ ba (aᵢ, cᵢ, sᵢ).", AMBER);
    // arrows between
    const ay = y0 + hh / 2;
    [MX + 2.58, MX + 5.58, MX + 9.58].forEach((ax) => {
      s.addShape(pres.shapes.LINE, { x: ax, y: ay, w: 0.36, h: 0, line: { color: MINT, width: 2.5, endArrowType: "triangle" } });
    });

    // stage 2 inner chain
    s.addText("CHUỖI XỬ LÝ BÊN TRONG GIAI ĐOẠN 2 (ACSD)", { x: MX, y: 4.4, w: 10, h: 0.3, margin: 0, fontFace: FS, fontSize: 11, bold: true, color: MINT, charSpacing: 1.5 });
    const chain = ["Khối mã hóa\n(Encoder)", "Token\nMerging", "Resize /\nCompact", "LCF\n(CDM/CDW)", "Hợp nhất\nđặc trưng", "Self-Attention\nAggregation", "2 đầu phân loại\ncategory + sentiment"];
    const cwid = (PW - 2 * MX - 6 * 0.28) / 7;
    chain.forEach((c, i) => {
      const x = MX + i * (cwid + 0.28), y = 4.8;
      const isKey = (i === 1 || i === 3);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cwid, h: 1.35, rectRadius: 0.09, fill: { color: isKey ? TEAL : NAVY2 }, line: { color: isKey ? MINT : "2E4A66", width: 1 } });
      s.addText(c, { x: x + 0.05, y, w: cwid - 0.1, h: 1.35, margin: 0, fontFace: FS, fontSize: 10.5, bold: isKey, color: isKey ? WHITE : "C9D6E2", align: "center", valign: "middle", lineSpacingMultiple: 1.0 });
      if (i < 6) s.addShape(pres.shapes.LINE, { x: x + cwid + 0.02, y: y + 0.67, w: 0.26, h: 0, line: { color: "6E8AA6", width: 1.5, endArrowType: "triangle" } });
    });
    s.addText("Hai khối then chốt (ToMe & LCF) được tô sáng — chính là đóng góp trọng tâm của LCTA.", { x: MX, y: 6.35, w: 11, h: 0.3, margin: 0, fontFace: FS, fontSize: 10.5, italic: true, color: "8FA6BC" });
    s.addNotes("Kiến trúc LCTA: tiền xử lý UOS → ATE (T5/GAS) → ACSD (encoder→ToMe→resize→LCF→fusion→SA→2 heads) → TASD.");
  }

  // ---------------- SLIDE 10 : Giai đoạn 1 ATE ----------------
  {
    const s = pres.addSlide();
    header(s, { kicker: "Giai đoạn 1", title: "Trích xuất cụm từ khía cạnh (ATE) bằng mô hình sinh", icon: ICON.extract, num: 10 });

    // left: approach
    card(s, MX, 1.6, 6.15, 2.4, LIGHT);
    s.addText("Sinh có điều kiện thay vì gán nhãn BIO", { x: MX + 0.3, y: 1.82, w: 5.55, h: 0.35, margin: 0, fontFace: FS, fontSize: 14, bold: true, color: INK });
    s.addText([
      { text: "Mô hình t5-base (tùy biến từ định dạng ", options: {} },
      { text: "GAS", options: { bold: true, color: TEAL } },
      { text: ") nhận câu và sinh trực tiếp danh sách cụm khía cạnh dưới dạng chuỗi có cấu trúc. Câu không có khía cạnh → sinh ra “none”.", options: {} },
    ], { x: MX + 0.3, y: 2.2, w: 5.55, h: 0.9, margin: 0, fontFace: FS, fontSize: 12.5, color: INK, lineSpacingMultiple: 1.15, valign: "top" });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.3, y: 3.2, w: 5.55, h: 0.62, rectRadius: 0.08, fill: { color: WHITE }, line: { color: LINEC, width: 1 } });
    s.addText([
      { text: "input:  ", options: { color: MUTE } },
      { text: "the room and staff were great", options: { color: INK } },
      { text: "\noutput:  ", options: { color: MUTE } },
      { text: "(room); (staff)", options: { color: TEAL, bold: true } },
    ], { x: MX + 0.45, y: 3.2, w: 5.3, h: 0.62, margin: 0, fontFace: "Courier New", fontSize: 10.5, valign: "middle", lineSpacingMultiple: 1.05 });

    // Levenshtein card
    card(s, MX, 4.2, 6.15, 2.35, WHITE);
    s.addImage({ data: ICON.shield, x: MX + 0.3, y: 4.5, w: 0.5, h: 0.5 });
    s.addText("Hậu xử lý bằng khoảng cách Levenshtein", { x: MX + 0.95, y: 4.5, w: 5.0, h: 0.5, margin: 0, fontFace: FS, fontSize: 13.5, bold: true, color: TEAL, valign: "middle" });
    s.addText("Đầu ra sinh có thể sai chính tả hoặc thừa/thiếu từ. Kỹ thuật đối sánh n-gram theo khoảng cách Levenshtein ánh xạ mỗi cụm sinh ra tới n-gram gần nhất thực sự xuất hiện trong câu → bảo đảm mọi khía cạnh là chuỗi con hợp lệ của văn bản gốc.", {
      x: MX + 0.3, y: 5.1, w: 5.55, h: 1.35, margin: 0, fontFace: FS, fontSize: 12, color: INK, lineSpacingMultiple: 1.18, valign: "top",
    });

    // right: result + error propagation
    card(s, 7.05, 1.6, PW - 7.05 - MX, 2.35, NAVY);
    s.addText("KẾT QUẢ ATE TRÊN TẬP TEST (GAS-ATE)", { x: 7.35, y: 1.85, w: 5, h: 0.3, margin: 0, fontFace: FS, fontSize: 10.5, bold: true, color: MINT, charSpacing: 1 });
    const stat = [["Precision", "80.26%"], ["Recall", "79.49%"], ["F1", "79.87%"]];
    stat.forEach((st, i) => {
      const x = 7.35 + i * ((PW - 7.05 - MX - 0.6) / 3 + 0.02), w = (PW - 7.05 - MX - 0.6) / 3 - 0.1;
      s.addText(st[1], { x, y: 2.25, w, h: 0.75, margin: 0, fontFace: FH, fontSize: 33, bold: true, color: i === 2 ? AMBER : WHITE, align: "center" });
      s.addText(st[0], { x, y: 3.05, w, h: 0.3, margin: 0, fontFace: FS, fontSize: 12, color: "AEC2D4", align: "center" });
    });
    s.addText("Precision > Recall: mô hình dự đoán thận trọng, sai ít nhưng vẫn bỏ sót một số aspect.", { x: 7.35, y: 3.4, w: PW - 7.05 - MX - 0.6, h: 0.45, margin: 0, fontFace: FS, fontSize: 10.5, italic: true, color: "8FA6BC", align: "center", valign: "top" });

    card(s, 7.05, 4.2, PW - 7.05 - MX, 2.35, LIGHT);
    s.addImage({ data: ICON.warning, x: 7.35, y: 4.5, w: 0.5, h: 0.5 });
    s.addText("Lan truyền lỗi (Error Propagation)", { x: 8.0, y: 4.5, w: 4.5, h: 0.5, margin: 0, fontFace: FS, fontSize: 13.5, bold: true, color: CORAL, valign: "middle" });
    s.addText("Recall chưa cao → một số aspect bị bỏ sót ở giai đoạn 1 sẽ không thể được phân loại đúng ở giai đoạn 2. Chất lượng bộ trích xuất là yếu tố quyết định của pipeline hai giai đoạn (phân tích định lượng ở Chương 4).", {
      x: 7.35, y: 5.1, w: PW - 7.05 - MX - 0.6, h: 1.35, margin: 0, fontFace: FS, fontSize: 12, color: INK, lineSpacingMultiple: 1.18, valign: "top",
    });
    s.addNotes("ATE: GAS/T5 sinh danh sách aspect, Levenshtein chuẩn hóa. F1=79.87%. Recall thấp gây error propagation.");
  }

  // ---------------- SLIDE 11 : Giai đoạn 2 ACSD pipeline ----------------
  {
    const s = pres.addSlide();
    header(s, { kicker: "Giai đoạn 2", title: "Bộ phân loại ACSD đa nhiệm: từ token đến bộ ba", icon: ICON.cogs, num: 11 });

    // vertical stacked steps with description
    const steps = [
      ["Đầu vào", "Định dạng cặp câu (SPC) + véc-tơ chỉ thị khía cạnh v đánh dấu vị trí aspect."],
      ["Khối mã hóa", "Backbone BERT / T5 encoder tạo biểu diễn ngữ cảnh H ∈ ℝ^(L×d)."],
      ["Token Merging", "Áp dụng BiToMe / SLM / SCM để hợp nhất token dư thừa, bảo vệ token khía cạnh & CLS/SEP."],
      ["Resize / Compact", "Nội suy chuỗi về độ dài gốc L (chế độ chính) hoặc giữ chuỗi rút gọn."],
      ["LCF (CDM / CDW)", "Làm nổi bật vùng ngữ cảnh cục bộ quanh khía cạnh trên chuỗi đã rút gọn."],
      ["Hợp nhất đặc trưng + Self-Attention", "Kết hợp luồng cục bộ với luồng toàn cục, tổng hợp tự chú ý trước phân loại."],
    ];
    const chh = 0.72, sx = MX, sw = 7.2;
    steps.forEach((st, i) => {
      const y = 1.6 + i * (chh + 0.15);
      s.addShape(pres.shapes.OVAL, { x: sx, y: y + 0.06, w: 0.6, h: 0.6, fill: { color: (i === 2 || i === 4) ? TEAL : NAVY }, line: { type: "none" }, shadow: shSoft() });
      s.addText(String(i + 1), { x: sx, y: y + 0.06, w: 0.6, h: 0.6, margin: 0, fontFace: FH, fontSize: 18, bold: true, color: WHITE, align: "center", valign: "middle" });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: sx + 0.78, y, w: sw - 0.78, h: chh, rectRadius: 0.08, fill: { color: (i === 2 || i === 4) ? LIGHT : WHITE }, line: { color: (i === 2 || i === 4) ? TEAL : LINEC, width: 1 } });
      s.addText([
        { text: st[0] + "   ", options: { bold: true, color: (i === 2 || i === 4) ? TEAL : INK } },
        { text: st[1], options: { color: MUTE } },
      ], { x: sx + 0.95, y, w: sw - 1.1, h: chh, margin: 0, fontFace: FS, fontSize: 11.5, valign: "middle", lineSpacingMultiple: 1.05 });
      if (i < steps.length - 1) s.addShape(pres.shapes.LINE, { x: sx + 0.3, y: y + chh + 0.02, w: 0, h: 0.13, line: { color: LINEC, width: 2 } });
    });

    // right: two output heads
    card(s, 8.15, 1.6, PW - 8.15 - MX, 4.95, NAVY);
    s.addText("HAI ĐẦU PHÂN LOẠI (MULTI-TASK)", { x: 8.45, y: 1.85, w: 4, h: 0.3, margin: 0, fontFace: FS, fontSize: 10.5, bold: true, color: MINT, charSpacing: 1 });

    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.45, y: 2.25, w: PW - 8.15 - MX - 0.6, h: 1.55, rectRadius: 0.1, fill: { color: NAVY2 }, line: { color: TEAL, width: 1.5 } });
    s.addText("Nhóm khía cạnh · 6 lớp", { x: 8.7, y: 2.45, w: 4, h: 0.35, margin: 0, fontFace: FS, fontSize: 13, bold: true, color: WHITE });
    s.addText("SERVICE · FACILITY · AMENITY · EXPERIENCE · BRANDING · LOYALTY", { x: 8.7, y: 2.85, w: PW - 8.15 - MX - 1.1, h: 0.85, margin: 0, fontFace: FS, fontSize: 11.5, color: "C9D6E2", lineSpacingMultiple: 1.25, valign: "top" });

    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.45, y: 4.0, w: PW - 8.15 - MX - 0.6, h: 1.35, rectRadius: 0.1, fill: { color: NAVY2 }, line: { color: MINT, width: 1.5 } });
    s.addText("Cảm xúc · 3 lớp", { x: 8.7, y: 4.2, w: 4, h: 0.35, margin: 0, fontFace: FS, fontSize: 13, bold: true, color: WHITE });
    s.addText([
      { text: "positive", options: { color: MINT, bold: true } },
      { text: "   ·   ", options: { color: "6E8AA6" } },
      { text: "negative", options: { color: CORAL, bold: true } },
      { text: "   ·   ", options: { color: "6E8AA6" } },
      { text: "neutral", options: { color: "AEC2D4", bold: true } },
    ], { x: 8.7, y: 4.6, w: PW - 8.15 - MX - 1.1, h: 0.6, margin: 0, fontFace: FS, fontSize: 13, valign: "middle" });

    s.addText("Chia sẻ tham số encoder; hàm mất mát tổng hợp cân bằng trọng số giữa hai nhiệm vụ.", { x: 8.45, y: 5.5, w: PW - 8.15 - MX - 0.6, h: 0.9, margin: 0, fontFace: FS, fontSize: 11, italic: true, color: "8FA6BC", lineSpacingMultiple: 1.15, valign: "top" });
    s.addNotes("Giai đoạn 2: chuỗi encoder→ToMe→resize→LCF→fusion→SA→2 heads (category 6 lớp, sentiment 3 lớp), học đa nhiệm.");
  }

  // ---------------- SLIDE 12 : BiToMe ----------------
  {
    const s = pres.addSlide();
    header(s, { kicker: "Chiến lược hợp nhất · 1/3", title: "BiToMe — Ghép cặp hai phía (Bipartite Token Merging)", icon: ICON.exchange, num: 12 });

    // left steps
    s.addText("Quy trình 5 bước", { x: MX, y: 1.55, w: 5, h: 0.35, margin: 0, fontFace: FS, fontSize: 14, bold: true, color: TEAL });
    const st = [
      ["Mã hóa đầu vào", "Câu + vị trí khía cạnh → H, véc-tơ chỉ thị v, mặt nạ token hợp lệ."],
      ["Xây tập ứng viên", "Loại [CLS], [SEP] và toàn bộ token khía cạnh (vᵢ > 0.5) để bảo vệ."],
      ["Chia hai nhóm A / B", "Sắp theo chỉ số tăng dần rồi chia xen kẽ chẵn–lẻ."],
      ["Tính cosine & ghép cặp", "Mỗi token A ghép token B tương đồng nhất — tham lam, một–một phía B."],
      ["Hợp nhất token", "Biểu diễn mới = trung bình cộng; nhãn bảo vệ v = max(vₐ, v_b)."],
    ];
    let yy = 1.98;
    st.forEach((x, i) => {
      s.addShape(pres.shapes.OVAL, { x: MX, y: yy, w: 0.5, h: 0.5, fill: { color: TEAL }, line: { type: "none" } });
      s.addText(String(i + 1), { x: MX, y: yy, w: 0.5, h: 0.5, margin: 0, fontFace: FH, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle" });
      s.addText([{ text: x[0] + ".  ", options: { bold: true, color: INK } }, { text: x[1], options: { color: MUTE } }], { x: MX + 0.68, y: yy - 0.04, w: 5.6, h: 0.62, margin: 0, fontFace: FS, fontSize: 11.8, lineSpacingMultiple: 1.08, valign: "middle" });
      yy += 0.7;
    });

    // right: A/B matching visual
    card(s, 6.9, 1.6, PW - 6.9 - MX, 3.15, LIGHT);
    s.addText("Ghép cặp hai phía trên ma trận độ tương đồng", { x: 7.2, y: 1.8, w: 5.5, h: 0.35, margin: 0, fontFace: FS, fontSize: 12.5, bold: true, color: INK });
    // group A
    const ax = 7.35, bx = 10.35, ty = 2.35, tw = 1.1, thh = 0.42, tg = 0.16;
    s.addText("Nhóm A", { x: ax, y: 2.1, w: tw, h: 0.25, margin: 0, fontFace: FS, fontSize: 10, bold: true, color: TEAL, align: "center" });
    s.addText("Nhóm B", { x: bx, y: 2.1, w: tw, h: 0.25, margin: 0, fontFace: FS, fontSize: 10, bold: true, color: TEAL_D, align: "center" });
    const A = ["p0", "p2", "p4"], B = ["p1", "p3", "p5"];
    A.forEach((t, i) => { s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: ax, y: ty + i * (thh + tg), w: tw, h: thh, rectRadius: 0.08, fill: { color: TEAL }, line: { type: "none" } }); s.addText(t, { x: ax, y: ty + i * (thh + tg), w: tw, h: thh, margin: 0, fontFace: FS, fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle" }); });
    B.forEach((t, i) => { s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: ty + i * (thh + tg), w: tw, h: thh, rectRadius: 0.08, fill: { color: TEAL_D }, line: { type: "none" } }); s.addText(t, { x: bx, y: ty + i * (thh + tg), w: tw, h: thh, margin: 0, fontFace: FS, fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle" }); });
    // matching lines
    [[0, 1], [1, 0], [2, 2]].forEach(([a, b]) => {
      s.addShape(pres.shapes.LINE, { x: ax + tw, y: ty + a * (thh + tg) + thh / 2, w: bx - (ax + tw), h: (b - a) * (thh + tg), line: { color: MINT, width: 2, endArrowType: "triangle" } });
    });
    s.addText("S = Â·B̂ᵀ  ·  chi phí O(|P|²/4 · H) ≪ O(L²·H) của self-attention.", { x: 7.2, y: 4.25, w: 5.5, h: 0.4, margin: 0, fontFace: FS, fontSize: 10.5, italic: true, color: MUTE, align: "center" });

    // pros/cons
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.9, y: 4.95, w: (PW - 6.9 - MX - 0.25) / 2, h: 1.55, rectRadius: 0.1, fill: { color: WHITE }, line: { color: MINT, width: 1.2 } });
    s.addImage({ data: ICON.check, x: 7.1, y: 5.15, w: 0.32, h: 0.32 });
    s.addText("Ưu điểm", { x: 7.5, y: 5.13, w: 2.2, h: 0.35, margin: 0, fontFace: FS, fontSize: 12.5, bold: true, color: TEAL, valign: "middle" });
    s.addText("Cặp merge nhất quán ngữ nghĩa cao, không chỉ dựa vị trí lân cận. Ổn định nhất giữa hai backbone.", { x: 7.1, y: 5.5, w: (PW - 6.9 - MX - 0.25) / 2 - 0.4, h: 0.95, margin: 0, fontFace: FS, fontSize: 11, color: INK, lineSpacingMultiple: 1.1, valign: "top" });
    const cx2 = 6.9 + (PW - 6.9 - MX - 0.25) / 2 + 0.25;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx2, y: 4.95, w: (PW - 6.9 - MX - 0.25) / 2, h: 1.55, rectRadius: 0.1, fill: { color: WHITE }, line: { color: CORAL, width: 1.2 } });
    s.addImage({ data: ICON.warning, x: cx2 + 0.2, y: 5.15, w: 0.32, h: 0.32 });
    s.addText("Nhược điểm", { x: cx2 + 0.6, y: 5.13, w: 2.5, h: 0.35, margin: 0, fontFace: FS, fontSize: 12.5, bold: true, color: CORAL, valign: "middle" });
    s.addText("Xây ma trận độ tương đồng làm tăng chi phí tính toán bổ sung.", { x: cx2 + 0.2, y: 5.5, w: (PW - 6.9 - MX - 0.25) / 2 - 0.4, h: 0.95, margin: 0, fontFace: FS, fontSize: 11, color: INK, lineSpacingMultiple: 1.1, valign: "top" });
    s.addNotes("BiToMe: chia A/B xen kẽ, ghép cặp tham lam theo cosine, bảo vệ aspect/CLS/SEP. Ổn định nhất.");
  }

  // ---------------- SLIDE 13 : SLM ----------------
  {
    const s = pres.addSlide();
    header(s, { kicker: "Chiến lược hợp nhất · 2/3", title: "SLM — Hợp nhất cục bộ tuần tự (Sequential Local Merging)", icon: ICON.arrows, num: 13 });

    // concept
    card(s, MX, 1.6, PW - 2 * MX, 1.5, LIGHT);
    s.addText([
      { text: "Nguyên lý:  ", options: { bold: true, color: TEAL } },
      { text: "quét tuần tự từ trái sang phải; mỗi token hợp nhất với ", options: { color: INK } },
      { text: "một trong hai token lân cận", options: { bold: true, color: INK } },
      { text: " (trái/phải) có độ tương đồng cosine cao hơn — bảo toàn thứ tự chuỗi và tính liên tục ngữ cảnh.", options: { color: INK } },
    ], { x: MX + 0.35, y: 1.75, w: PW - 2 * MX - 0.7, h: 0.65, margin: 0, fontFace: FS, fontSize: 13.5, lineSpacingMultiple: 1.15, valign: "middle" });
    s.addText([
      { text: "Bảo vệ ranh giới câu–khía cạnh:  ", options: { bold: true, color: TEAL_D } },
      { text: "Boundary(j)=1[vⱼ<0.5 ∧ vⱼ₊₁>0.5] tránh hợp nhất token thuộc hai phần khác nhau của định dạng SPC.", options: { color: MUTE } },
    ], { x: MX + 0.35, y: 2.42, w: PW - 2 * MX - 0.7, h: 0.55, margin: 0, fontFace: FS, fontSize: 11.5, lineSpacingMultiple: 1.1, valign: "middle" });

    // cascade example visual
    card(s, MX, 3.35, 7.0, 3.2, WHITE);
    s.addText("Ví dụ: giảm 7 → 5 token", { x: MX + 0.3, y: 3.55, w: 6, h: 0.35, margin: 0, fontFace: FS, fontSize: 13, bold: true, color: INK });
    const toks = ["the", "service", "was", "was", "tasty", "and", "good"];
    let tx = MX + 0.35;
    toks.forEach((t, i) => {
      const col = (i === 1 || i === 2 || i === 3 || i === 4) ? MINT : TEAL;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: tx, y: 4.0, w: 0.85, h: 0.45, rectRadius: 0.08, fill: { color: col }, line: { type: "none" } });
      s.addText(t, { x: tx, y: 4.0, w: 0.85, h: 0.45, margin: 0, fontFace: FS, fontSize: 10, bold: true, color: WHITE, align: "center", valign: "middle" });
      tx += 0.93;
    });
    s.addText("↓  hợp nhất (service+was), (was+tasty)", { x: MX + 0.35, y: 4.6, w: 6, h: 0.35, margin: 0, fontFace: FS, fontSize: 11, italic: true, color: MUTE });
    const toks2 = [["the", TEAL], ["service·was", MINT], ["was·tasty", MINT], ["and", TEAL], ["good", TEAL]];
    tx = MX + 0.35;
    toks2.forEach(([t, col]) => {
      const ww = t.includes("·") ? 1.35 : 0.85;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: tx, y: 5.05, w: ww, h: 0.45, rectRadius: 0.08, fill: { color: col }, line: { type: "none" } });
      s.addText(t, { x: tx, y: 5.05, w: ww, h: 0.45, margin: 0, fontFace: FS, fontSize: 9.5, bold: true, color: WHITE, align: "center", valign: "middle" });
      tx += ww + 0.1;
    });
    s.addText("Không ràng buộc một–một → một token có thể liên tiếp hấp thụ nhiều lân cận (cascade merging), rút gọn nhanh vùng tương đồng cao.", {
      x: MX + 0.3, y: 5.7, w: 6.4, h: 0.75, margin: 0, fontFace: FS, fontSize: 11, color: INK, lineSpacingMultiple: 1.12, valign: "top",
    });

    // pros/cons right
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.75, y: 3.35, w: PW - 7.75 - MX, h: 1.5, rectRadius: 0.1, fill: { color: LIGHT }, line: { color: MINT, width: 1.2 } });
    s.addImage({ data: ICON.check, x: 8.0, y: 3.58, w: 0.34, h: 0.34 });
    s.addText("Ưu điểm", { x: 8.45, y: 3.56, w: 3, h: 0.35, margin: 0, fontFace: FS, fontSize: 13, bold: true, color: TEAL, valign: "middle" });
    s.addText("Bảo toàn thứ tự & liên tục ngữ cảnh; hoạt động tốt trên BERT — đạt Micro-F1 cao nhất ở TASD (71.18%).", { x: 8.0, y: 3.95, w: PW - 7.75 - MX - 0.5, h: 0.85, margin: 0, fontFace: FS, fontSize: 11.3, color: INK, lineSpacingMultiple: 1.12, valign: "top" });

    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.75, y: 5.05, w: PW - 7.75 - MX, h: 1.5, rectRadius: 0.1, fill: { color: LIGHT }, line: { color: CORAL, width: 1.2 } });
    s.addImage({ data: ICON.warning, x: 8.0, y: 5.28, w: 0.34, h: 0.34 });
    s.addText("Nhược điểm", { x: 8.45, y: 5.26, w: 3, h: 0.35, margin: 0, fontFace: FS, fontSize: 13, bold: true, color: CORAL, valign: "middle" });
    s.addText("Chỉ xét lân cận → cặp hợp nhất chưa tối ưu khi token tương đồng cao ở xa nhau; phụ thuộc nhiều vào backbone (yếu trên T5).", { x: 8.0, y: 5.65, w: PW - 7.75 - MX - 0.5, h: 0.85, margin: 0, fontFace: FS, fontSize: 11.3, color: INK, lineSpacingMultiple: 1.12, valign: "top" });
    s.addNotes("SLM: quét trái→phải, gộp lân cận cosine cao hơn, cascade merging. Tốt trên BERT, yếu trên T5.");
  }

  // ---------------- SLIDE 14 : SCM + so sánh ----------------
  {
    const s = pres.addSlide();
    header(s, { kicker: "Chiến lược hợp nhất · 3/3", title: "SCM & so sánh ba chiến lược Token Merging", icon: ICON.route, num: 14 });

    // SCM concept
    card(s, MX, 1.6, 5.6, 2.05, LIGHT);
    s.addText("SCM · Sequential Cosine Merging", { x: MX + 0.3, y: 1.8, w: 5.0, h: 0.35, margin: 0, fontFace: FS, fontSize: 13.5, bold: true, color: TEAL });
    s.addText([
      { text: "Duyệt token trái→phải chọn token trái nhất làm nguồn, rồi hợp nhất với token có cosine cao nhất ", options: { color: INK } },
      { text: "trong toàn chuỗi", options: { bold: true, color: TEAL } },
      { text: " (không giới hạn lân cận). Bảo vệ token khía cạnh; [CLS]/[SEP] có thể nhận nhưng không bao giờ làm nguồn.", options: { color: INK } },
    ], { x: MX + 0.3, y: 2.15, w: 5.0, h: 1.4, margin: 0, fontFace: FS, fontSize: 11.8, lineSpacingMultiple: 1.15, valign: "top" });

    // SCM visual
    card(s, MX, 3.8, 5.6, 2.7, WHITE);
    s.addText("Ví dụ: gộp “but” theo cosine toàn chuỗi", { x: MX + 0.3, y: 4.0, w: 5, h: 0.35, margin: 0, fontFace: FS, fontSize: 12.5, bold: true, color: INK });
    const sctoks = [["food", AMBER], ["is", TEAL], ["tasty", TEAL], ["but", CORAL], ["poor", TEAL]];
    let tx = MX + 0.35;
    sctoks.forEach(([t, col]) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: tx, y: 4.45, w: 0.95, h: 0.5, rectRadius: 0.08, fill: { color: col }, line: { type: "none" } });
      s.addText(t, { x: tx, y: 4.45, w: 0.95, h: 0.5, margin: 0, fontFace: FS, fontSize: 10.5, bold: true, color: WHITE, align: "center", valign: "middle" });
      tx += 1.02;
    });
    s.addText([
      { text: "■ ", options: { color: AMBER, bold: true } },
      { text: "aspect term (bảo vệ)    ", options: { color: MUTE } },
      { text: "■ ", options: { color: CORAL, bold: true } },
      { text: "token bị hợp nhất", options: { color: MUTE } },
    ], { x: MX + 0.35, y: 5.15, w: 5, h: 0.3, margin: 0, fontFace: FS, fontSize: 10 });
    s.addText("Tìm token đích trên toàn chuỗi → gộp được token tương đồng dù ở xa; bù lại thứ tự duyệt cố định chưa hẳn tối ưu về ngữ nghĩa.", {
      x: MX + 0.3, y: 5.5, w: 5.0, h: 0.9, margin: 0, fontFace: FS, fontSize: 11, color: INK, lineSpacingMultiple: 1.12, valign: "top",
    });

    // comparison table right
    s.addText("SO SÁNH BA CHIẾN LƯỢC THEO TIÊU CHÍ", { x: 6.5, y: 1.6, w: 6, h: 0.3, margin: 0, fontFace: FS, fontSize: 11, bold: true, color: MUTE, charSpacing: 1.5 });
    const hdr = (t, c) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: c }, align: "center" } });
    const cmp = [
      [{ text: "Tiêu chí", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "left" } }, hdr("BiToMe", TEAL), hdr("SLM", MINT), hdr("SCM", TEAL_D)],
      ["Ổn định giữa 2 backbone", { text: "Cao", options: { bold: true, color: TEAL } }, "Thấp", "Trung bình"],
      ["Hiệu năng trên BERT", "Tốt, cân bằng", "Micro-F1 cao nhất", { text: "Macro-F1 cao nhất", options: { bold: true, color: TEAL_D } }],
      ["Hiệu năng trên T5", { text: "Đạt đỉnh cả 3 chỉ số", options: { bold: true, color: TEAL } }, "Giảm", "Giảm rõ rệt"],
      ["Thế mạnh chỉ số", "Cân bằng Micro/Macro", "Micro-F1", "Macro-F1 (lớp hiếm)"],
      ["Phụ thuộc backbone", "Thấp", "Cao", "Trung bình–cao"],
      ["Phụ thuộc chất lượng aspect", "Thấp", "Thấp", { text: "Cao", options: { color: CORAL } }],
      ["Khả năng khái quát hóa", { text: "Cao", options: { bold: true, color: TEAL } }, "Hạn chế", "Phụ thuộc đầu vào"],
    ];
    s.addTable(cmp, {
      x: 6.5, y: 1.95, w: PW - 6.5 - MX, colW: [1.95, 1.55, 1.35, 1.36],
      rowH: [0.44, 0.52, 0.52, 0.52, 0.52, 0.44, 0.5, 0.5],
      fontFace: FS, fontSize: 10.3, color: INK, valign: "middle", align: "center",
      border: { pt: 1, color: LINEC }, fill: { color: WHITE },
    });
    s.addNotes("SCM: nguồn trái nhất, đích cosine toàn chuỗi. Bảng so sánh BiToMe (ổn định), SLM (Micro trên BERT), SCM (Macro trên BERT).");
  }

  // ---------------- SLIDE 15 : LLM tách câu ----------------
  {
    const s = pres.addSlide();
    header(s, { kicker: "Tiền xử lý", title: "Tách đơn vị ý kiến (UOS) bằng mô hình ngôn ngữ lớn", icon: ICON.scissors, num: 15 });

    // left: motivation + rules + example
    card(s, MX, 1.6, 6.5, 2.05, LIGHT);
    s.addText([
      { text: "Động cơ:  ", options: { bold: true, color: TEAL } },
      { text: "câu review dài chứa nhiều khía cạnh làm biểu diễn từng khía cạnh bị pha trộn → dễ bỏ sót. LLM ", options: { color: INK } },
      { text: "Qwen3-8B", options: { bold: true, color: INK } },
      { text: " tách câu thành các UOS độc lập, mỗi UOS tập trung một khía cạnh, theo quy tắc ngôn ngữ (không phân loại cảm xúc).", options: { color: INK } },
    ], { x: MX + 0.3, y: 1.78, w: 5.9, h: 1.7, margin: 0, fontFace: FS, fontSize: 12.5, lineSpacingMultiple: 1.18, valign: "top" });

    // example transformation
    card(s, MX, 3.8, 6.5, 2.7, WHITE);
    s.addText("Ví dụ  1 → 4 UOS", { x: MX + 0.3, y: 4.0, w: 5, h: 0.35, margin: 0, fontFace: FS, fontSize: 12.5, bold: true, color: TEAL });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.3, y: 4.38, w: 5.9, h: 0.6, rectRadius: 0.08, fill: { color: LIGHT2 }, line: { type: "none" } });
    s.addText("“the quality of the restaurant, service, rooms as well as the staff all bring a great feeling”", { x: MX + 0.45, y: 4.38, w: 5.6, h: 0.6, margin: 0, fontFace: FS, fontSize: 10.5, italic: true, color: INK, valign: "middle" });
    s.addText("↓", { x: MX + 0.3, y: 4.98, w: 5.9, h: 0.25, margin: 0, fontFace: FS, fontSize: 13, bold: true, color: TEAL, align: "center" });
    const uos = ["restaurant is great", "service is great", "rooms are great", "staff is great"];
    uos.forEach((u, i) => {
      const x = MX + 0.3 + (i % 2) * 3.0, y = 5.28 + Math.floor(i / 2) * 0.55;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 2.85, h: 0.45, rectRadius: 0.08, fill: { color: WHITE }, line: { color: TEAL, width: 1 } });
      s.addText([{ text: "(" + (i + 1) + ") ", options: { bold: true, color: TEAL } }, { text: "The " + u + ".", options: { color: INK } }], { x: x + 0.1, y, w: 2.65, h: 0.45, margin: 0, fontFace: FS, fontSize: 9.8, valign: "middle" });
    });

    // right: statistics
    card(s, 7.4, 1.6, PW - 7.4 - MX, 2.55, NAVY);
    s.addText("THỐNG KÊ TÁCH TRÊN TẬP KIỂM TRA", { x: 7.7, y: 1.85, w: 5, h: 0.3, margin: 0, fontFace: FS, fontSize: 10.5, bold: true, color: MINT, charSpacing: 1 });
    const stt = [["312", "câu đầu vào"], ["612", "UOS đầu ra"], ["1.96", "UOS / câu"], ["55.8%", "câu được tách"]];
    stt.forEach((v, i) => {
      const x = 7.7 + (i % 2) * ((PW - 7.4 - MX - 0.6) / 2), y = 2.25 + Math.floor(i / 2) * 0.95;
      s.addText(v[0], { x, y, w: (PW - 7.4 - MX - 0.6) / 2, h: 0.6, margin: 0, fontFace: FH, fontSize: 30, bold: true, color: i === 3 ? AMBER : WHITE, align: "left" });
      s.addText(v[1], { x, y: y + 0.58, w: (PW - 7.4 - MX - 0.6) / 2, h: 0.3, margin: 0, fontFace: FS, fontSize: 11, color: "AEC2D4" });
    });

    // effect chart (recall improvement)
    card(s, 7.4, 4.3, PW - 7.4 - MX, 2.2, WHITE);
    s.addText("Cải thiện Micro-Recall sau khi tách câu (%)", { x: 7.7, y: 4.45, w: 5, h: 0.3, margin: 0, fontFace: FS, fontSize: 11.5, bold: true, color: INK });
    s.addChart(pres.charts.BAR, [
      { name: "No-split", labels: ["Baseline", "BiToMe+CDM (T5)", "SLM+CDM (BERT)"], values: [70.19, 69.23, 70.83] },
      { name: "+ Tách câu", labels: ["Baseline", "BiToMe+CDM (T5)", "SLM+CDM (BERT)"], values: [71.15, 71.47, 72.12] },
    ], {
      x: 7.55, y: 4.75, w: PW - 7.4 - MX - 0.3, h: 1.65, barDir: "col",
      chartColors: [LINEC, TEAL], showValue: false, valAxisMinVal: 66, valAxisMaxVal: 74,
      catAxisLabelColor: MUTE, valAxisLabelColor: MUTE, catAxisLabelFontSize: 8, valAxisLabelFontSize: 8,
      valGridLine: { color: LIGHT2, size: 0.5 }, catGridLine: { style: "none" },
      showLegend: true, legendPos: "t", legendFontSize: 8, legendColor: MUTE,
      chartArea: { fill: { color: WHITE } },
    });
    s.addNotes("UOS tách câu bằng Qwen3-8B. 312→612 UOS, 55.8% câu được tách. Recall cải thiện 1-2pp, T5 hưởng lợi nhiều nhất (+2.24pp).");
  }

  // ---------------- SLIDE 16 : Dữ liệu ----------------
  {
    const s = pres.addSlide();
    header(s, { kicker: "Thực nghiệm", title: "Tập dữ liệu đánh giá khách sạn & mất cân bằng nhãn", icon: ICON.database, num: 16 });

    // key stats row
    const stat = [["2 934", "câu review", TEAL], ["6", "nhóm khía cạnh", TEAL_D], ["3", "nhãn cảm xúc", MINT], ["80:10:10", "train/dev/test", AMBER]];
    const sw = (PW - 2 * MX - 3 * 0.25) / 4;
    stat.forEach((st, i) => {
      const x = MX + i * (sw + 0.25);
      card(s, x, 1.6, sw, 1.15, LIGHT);
      s.addText(st[0], { x: x + 0.15, y: 1.72, w: sw - 0.3, h: 0.6, margin: 0, fontFace: FH, fontSize: 28, bold: true, color: st[2], align: "left" });
      s.addText(st[1], { x: x + 0.15, y: 2.34, w: sw - 0.3, h: 0.3, margin: 0, fontFace: FS, fontSize: 11, color: MUTE });
    });

    // aspect categories left
    card(s, MX, 2.95, 6.0, 3.55, WHITE);
    s.addText("Sáu nhóm khía cạnh (miền khách sạn)", { x: MX + 0.3, y: 3.15, w: 5.4, h: 0.35, margin: 0, fontFace: FS, fontSize: 13.5, bold: true, color: TEAL });
    const asp = [
      ["FACILITY", "Cơ sở vật chất, phòng, kiến trúc, hồ bơi…"],
      ["SERVICE", "Chất lượng món ăn, quy trình, thái độ nhân viên"],
      ["AMENITY", "Tiện ích: bãi xe, spa, nhà hàng, an ninh…"],
      ["EXPERIENCE", "Cảm nhận tổng thể, không khí, mức thư giãn"],
      ["BRANDING", "Hình ảnh, uy tín, giá trị thương hiệu"],
      ["LOYALTY", "Ý định quay lại, giới thiệu cho người khác"],
    ];
    let yy = 3.55;
    asp.forEach(([a, d], i) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.3, y: yy, w: 1.6, h: 0.4, rectRadius: 0.08, fill: { color: i < 3 ? TEAL : (i < 5 ? TEAL_D : CORAL) }, line: { type: "none" } });
      s.addText(a, { x: MX + 0.3, y: yy, w: 1.6, h: 0.4, margin: 0, fontFace: FS, fontSize: 10, bold: true, color: WHITE, align: "center", valign: "middle" });
      s.addText(d, { x: MX + 2.05, y: yy, w: 3.65, h: 0.4, margin: 0, fontFace: FS, fontSize: 10.5, color: INK, valign: "middle" });
      yy += 0.48;
    });

    // imbalance chart right
    card(s, 7.05, 2.95, PW - 7.05 - MX, 3.55, WHITE);
    s.addText("Phân bố nhãn cảm xúc (tập Train) — mất cân bằng nghiêm trọng", { x: 7.35, y: 3.12, w: 5.5, h: 0.55, margin: 0, fontFace: FS, fontSize: 12.5, bold: true, color: INK, lineSpacingMultiple: 1.05 });
    s.addChart(pres.charts.DOUGHNUT, [{
      name: "Sentiment", labels: ["Positive", "Negative", "Neutral"], values: [91.4, 6.9, 1.8],
    }], {
      x: 7.2, y: 3.7, w: 3.0, h: 2.65, holeSize: 55,
      chartColors: [MINT, CORAL, AMBER], showValue: false,
      showLegend: false, chartArea: { fill: { color: WHITE } },
    });
    // legend + note
    const lg = [["Positive", "91.4%", MINT], ["Negative", "6.9%", CORAL], ["Neutral", "1.8%", AMBER]];
    let ly = 3.95;
    lg.forEach(([n, v, c]) => {
      s.addShape(pres.shapes.OVAL, { x: 10.35, y: ly + 0.03, w: 0.2, h: 0.2, fill: { color: c }, line: { type: "none" } });
      s.addText([{ text: n + "  ", options: { color: INK } }, { text: v, options: { bold: true, color: c } }], { x: 10.65, y: ly - 0.05, w: 2.0, h: 0.35, margin: 0, fontFace: FS, fontSize: 12.5, valign: "middle" });
      ly += 0.5;
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 10.35, y: 5.5, w: PW - 10.35 - MX - 0.3, h: 0.85, rectRadius: 0.08, fill: { color: LIGHT }, line: { type: "none" } });
    s.addText("→ Macro-F1 trở nên quan trọng hơn Micro-F1. Áp dụng tăng cường dữ liệu cho lớp Negative & Neutral.", { x: 10.5, y: 5.55, w: PW - 10.35 - MX - 0.6, h: 0.75, margin: 0, fontFace: FS, fontSize: 10, color: TEAL_D, lineSpacingMultiple: 1.1, valign: "middle" });
    s.addNotes("Dữ liệu Booking.com khách sạn: 2934 câu, 6 aspect, 3 sentiment. Mất cân bằng: 91.4% positive. Tăng cường dữ liệu cho Neg/Neu.");
  }

  // ---------------- SLIDE 17 : Thiết kế thực nghiệm ----------------
  {
    const s = pres.addSlide();
    header(s, { kicker: "Thực nghiệm", title: "Thiết kế ablation & cài đặt siêu tham số", icon: ICON.flask, num: 17 });

    // 12 config matrix
    card(s, MX, 1.6, 6.4, 4.9, WHITE);
    s.addText("12 cấu hình = LCF × ToMe × chiến lược", { x: MX + 0.3, y: 1.8, w: 5.8, h: 0.35, margin: 0, fontFace: FS, fontSize: 13.5, bold: true, color: TEAL });
    s.addText("Sơ đồ ablation có hệ thống (chế độ resize) để tách riêng ảnh hưởng của từng yếu tố:", { x: MX + 0.3, y: 2.15, w: 5.8, h: 0.5, margin: 0, fontFace: FS, fontSize: 11, color: MUTE, lineSpacingMultiple: 1.1 });
    const cfg = [
      ["Baseline", "encoder", "không LCF · không ToMe (mốc)"],
      ["CDM / CDW", "chỉ LCF", "che cứng / trọng số mềm, chưa ToMe"],
      ["BiToMe / SLM / SCM", "chỉ ToMe", "ba chiến lược, không LCF"],
      ["BiToMe + CDM/CDW", "ToMe × LCF", "ghép cặp hai phía kết hợp LCF"],
      ["SLM + CDM/CDW", "ToMe × LCF", "cục bộ tuần tự kết hợp LCF"],
      ["SCM + CDM/CDW", "ToMe × LCF", "cosine toàn chuỗi kết hợp LCF"],
    ];
    let yy = 2.75;
    cfg.forEach((c, i) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.3, y: yy, w: 5.8, h: 0.55, rectRadius: 0.07, fill: { color: i % 2 ? LIGHT : LIGHT2 }, line: { type: "none" } });
      s.addText(c[0], { x: MX + 0.45, y: yy, w: 2.35, h: 0.55, margin: 0, fontFace: FS, fontSize: 11, bold: true, color: TEAL_D, valign: "middle" });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 2.85, y: yy + 0.13, w: 1.15, h: 0.3, rectRadius: 0.15, fill: { color: TEAL }, line: { type: "none" } });
      s.addText(c[1], { x: MX + 2.85, y: yy + 0.13, w: 1.15, h: 0.3, margin: 0, fontFace: FS, fontSize: 8.5, bold: true, color: WHITE, align: "center", valign: "middle" });
      s.addText(c[2], { x: MX + 4.1, y: yy, w: 2.0, h: 0.55, margin: 0, fontFace: FS, fontSize: 9.3, color: MUTE, valign: "middle", lineSpacingMultiple: 1.0 });
      yy += 0.62;
    });

    // hyperparams right
    card(s, 7.15, 1.6, PW - 7.15 - MX, 4.9, LIGHT);
    s.addImage({ data: ICON.cogs, x: 7.45, y: 1.85, w: 0.45, h: 0.45 });
    s.addText("Siêu tham số chung", { x: 8.0, y: 1.85, w: 4, h: 0.45, margin: 0, fontFace: FS, fontSize: 14, bold: true, color: INK, valign: "middle" });
    const hp = [
      ["Random seed", "42"], ["Số epoch tối đa", "15  (patience 4)"], ["Batch size", "16"],
      ["Learning rate", "2 × 10⁻⁵"], ["Độ dài chuỗi tối đa", "128 token"], ["Dropout", "0.10"],
      ["Attention head (SA)", "8"], ["SRD threshold α", "5"], ["Merge ToMe (post-BERT)", "2 bước"],
      ["Tiêu chí early stopping", "Dev TASD Micro-F1"],
    ];
    let hy = 2.45;
    hp.forEach(([k, v], i) => {
      if (i > 0) s.addShape(pres.shapes.LINE, { x: 7.45, y: hy, w: PW - 7.15 - MX - 0.6, h: 0, line: { color: LINEC, width: 0.75 } });
      s.addText(k, { x: 7.45, y: hy + 0.02, w: 3.0, h: 0.38, margin: 0, fontFace: FS, fontSize: 11.5, color: MUTE, valign: "middle" });
      s.addText(v, { x: 10.3, y: hy + 0.02, w: PW - 10.3 - MX - 0.2, h: 0.38, margin: 0, fontFace: FS, fontSize: 11.5, bold: true, color: INK, align: "right", valign: "middle" });
      hy += 0.4;
    });
    s.addNotes("12 cấu hình ablation. Hyperparams: seed 42, 15 epoch, batch 16, lr 2e-5, L=128, α=5, resize.");
  }

  // ---------------- SLIDE 18 : Kết quả ACSD ----------------
  {
    const s = pres.addSlide();
    header(s, { kicker: "Kết quả · ACSD (Oracle Aspect, BERT)", title: "Kết hợp ToMe + LCF cải thiện mạnh Macro-F1", icon: ICON.chartbar, num: 18 });

    // chart
    card(s, MX, 1.6, 7.5, 4.9, WHITE);
    s.addText("Macro-F1 (%) trên tập test — backbone BERT", { x: MX + 0.3, y: 1.78, w: 6, h: 0.3, margin: 0, fontFace: FS, fontSize: 12.5, bold: true, color: INK });
    s.addChart(pres.charts.BAR, [{
      name: "Macro-F1",
      labels: ["Baseline", "CDM", "CDW", "BiToMe", "SLM", "SCM", "BiToMe+CDM", "SLM+CDM", "SCM+CDM"],
      values: [63.75, 57.96, 61.01, 64.03, 62.95, 62.31, 68.73, 64.28, 71.98],
    }], {
      x: MX + 0.2, y: 2.15, w: 7.1, h: 4.15, barDir: "col",
      chartColors: [MUTE, CORAL, CORAL, TEAL, TEAL, TEAL, MINT, MINT, AMBER],
      showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 8.5, dataLabelFontBold: true,
      valAxisMinVal: 55, valAxisMaxVal: 75,
      catAxisLabelColor: MUTE, valAxisLabelColor: MUTE, catAxisLabelFontSize: 8.5, valAxisLabelFontSize: 8,
      valGridLine: { color: LIGHT2, size: 0.5 }, catGridLine: { style: "none" },
      showLegend: false, chartArea: { fill: { color: WHITE } }, catAxisLabelRotate: 30,
    });

    // right takeaways
    card(s, 8.35, 1.6, PW - 8.35 - MX, 2.35, NAVY);
    s.addImage({ data: ICON.trophyW, x: 8.65, y: 1.9, w: 0.5, h: 0.5 });
    s.addText("Cấu hình tốt nhất", { x: 9.25, y: 1.9, w: 3.5, h: 0.5, margin: 0, fontFace: FS, fontSize: 13, bold: true, color: WHITE, valign: "middle" });
    s.addText([{ text: "SCM + CDM", options: { bold: true, color: MINT, fontSize: 20, breakLine: true } }, { text: "Macro-F1 = ", options: { color: "AEC2D4", fontSize: 12 } }, { text: "71.98%", options: { bold: true, color: AMBER, fontSize: 20 } }], { x: 8.65, y: 2.5, w: 4.0, h: 0.75, margin: 0, fontFace: FH, valign: "middle" });
    s.addText("Cao hơn Baseline +8.23 điểm % — tốt nhất toàn bảng; xử lý tốt các lớp thiểu số.", { x: 8.65, y: 3.25, w: PW - 8.35 - MX - 0.6, h: 0.6, margin: 0, fontFace: FS, fontSize: 10.5, italic: true, color: "8FA6BC", lineSpacingMultiple: 1.1, valign: "top" });

    const findings = [
      ["ToMe + LCF > riêng lẻ", "Chỉ dùng LCF không cải thiện: CDM tụt xuống 57.96% (−5.79 so với Baseline)."],
      ["BiToMe + CDM", "68.73% (+4.98) — ổn định, cân bằng chi phí–hiệu năng."],
      ["Micro-F1 tương đương", "Nhiều cấu hình đạt 86.86%; khác biệt nằm ở Macro-F1 (lớp hiếm)."],
    ];
    let fy = 4.1;
    findings.forEach(([t, d]) => {
      s.addImage({ data: ICON.check, x: 8.35, y: fy + 0.02, w: 0.26, h: 0.26 });
      s.addText([{ text: t + ".  ", options: { bold: true, color: TEAL } }, { text: d, options: { color: INK } }], { x: 8.75, y: fy - 0.05, w: PW - 8.75 - MX, h: 0.85, margin: 0, fontFace: FS, fontSize: 11, lineSpacingMultiple: 1.1, valign: "top" });
      fy += 0.83;
    });
    s.addNotes("ACSD trên BERT: SCM+CDM 71.98% Macro-F1 (best, +8.23). BiToMe+CDM 68.73%. LCF đơn lẻ tệ (CDM 57.96%). ToMe chỉ hiệu quả khi kết hợp LCF.");
  }

  // ---------------- SLIDE 19 : Kết quả TASD + phân tích ----------------
  {
    const s = pres.addSlide();
    header(s, { kicker: "Kết quả · TASD & phân tích", title: "Hiệu quả phụ thuộc backbone · resize vượt compact", icon: ICON.chartbar, num: 19 });

    // TASD table
    card(s, MX, 1.6, 6.3, 3.15, WHITE);
    s.addText("TASD Micro-F1 (%) — pipeline đầu–cuối", { x: MX + 0.3, y: 1.78, w: 5.7, h: 0.3, margin: 0, fontFace: FS, fontSize: 12.5, bold: true, color: INK });
    const hdr = (t, c) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: c }, align: "center" } });
    const tasd = [
      [{ text: "Model", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "left" } }, hdr("BERT", TEAL), hdr("T5", TEAL_D)],
      ["Baseline", "70.53", "69.57"],
      ["LCTA-BiToMe-CDM", "69.24", { text: "71.18", options: { bold: true, color: TEAL_D } }],
      ["LCTA-SLM-CDM", { text: "71.18", options: { bold: true, color: TEAL } }, "69.89"],
      ["LCTA-SLM-CDW", "70.21", "69.89"],
      ["LCTA-SCM-CDM", "69.89", "67.95"],
      ["GAS (tham chiếu)", "—", "70.51"],
      ["TOFA (tham chiếu)", { text: "62.65", options: { color: CORAL } }, "—"],
    ];
    s.addTable(tasd, {
      x: MX + 0.3, y: 2.1, w: 5.7, colW: [3.1, 1.3, 1.3],
      rowH: [0.35, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31],
      fontFace: FS, fontSize: 10.5, color: INK, valign: "middle", align: "center",
      border: { pt: 1, color: LINEC }, fill: { color: WHITE },
    });

    // insight cards under table
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 4.95, w: 6.3, h: 1.55, rectRadius: 0.1, fill: { color: LIGHT }, line: { color: LINEC, width: 1 } });
    s.addImage({ data: ICON.brain, x: MX + 0.28, y: 5.2, w: 0.5, h: 0.5 });
    s.addText([
      { text: "Không có chiến lược tối ưu cho mọi backbone.  ", options: { bold: true, color: TEAL } },
      { text: "SLM-CDM tốt nhất trên BERT (71.18%); BiToMe-CDM tốt nhất trên T5 (71.18%, +1.61 vs Baseline). Baseline BERT (70.53%) đã vượt TOFA (+8.53) và nhỉnh hơn GAS.", options: { color: INK } },
    ], { x: MX + 0.95, y: 5.15, w: 5.15, h: 1.25, margin: 0, fontFace: FS, fontSize: 11.5, lineSpacingMultiple: 1.15, valign: "middle" });

    // right: resize vs compact + LLM
    card(s, 7.15, 1.6, PW - 7.15 - MX, 2.35, WHITE);
    s.addText("Resize vs Compact — Macro-F1 (%)", { x: 7.45, y: 1.78, w: 5, h: 0.3, margin: 0, fontFace: FS, fontSize: 12.5, bold: true, color: INK });
    s.addChart(pres.charts.BAR, [
      { name: "Resize", labels: ["SCM+CDM", "BiToMe+CDM", "SLM+CDM", "SLM+CDW"], values: [57.16, 54.20, 50.00, 50.00] },
      { name: "Compact", labels: ["SCM+CDM", "BiToMe+CDM", "SLM+CDM", "SLM+CDW"], values: [48.86, 45.91, 49.86, 49.57] },
    ], {
      x: 7.35, y: 2.1, w: PW - 7.15 - MX - 0.4, h: 1.75, barDir: "col",
      chartColors: [TEAL, LINEC], showValue: false, valAxisMinVal: 42, valAxisMaxVal: 60,
      catAxisLabelColor: MUTE, valAxisLabelColor: MUTE, catAxisLabelFontSize: 8, valAxisLabelFontSize: 7.5,
      valGridLine: { color: LIGHT2, size: 0.5 }, catGridLine: { style: "none" },
      showLegend: true, legendPos: "t", legendFontSize: 8, legendColor: MUTE,
      chartArea: { fill: { color: WHITE } },
    });

    // two takeaway strips
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.15, y: 4.15, w: PW - 7.15 - MX, h: 1.1, rectRadius: 0.1, fill: { color: NAVY }, line: { type: "none" } });
    s.addImage({ data: ICON.balanceW || ICON.balance, x: 7.4, y: 4.4, w: 0.45, h: 0.45 });
    s.addText([{ text: "Resize > Compact  ", options: { bold: true, color: MINT } }, { text: "trung bình +4.29 điểm % Macro-F1. Khôi phục độ dài token giúp bảo toàn ngữ nghĩa (đặc biệt SCM/BiToMe: +8 điểm).", options: { color: "D5E1EC" } }], { x: 7.95, y: 4.32, w: PW - 7.15 - MX - 1.0, h: 0.85, margin: 0, fontFace: FS, fontSize: 11, lineSpacingMultiple: 1.12, valign: "middle" });

    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.15, y: 5.4, w: PW - 7.15 - MX, h: 1.1, rectRadius: 0.1, fill: { color: LIGHT }, line: { color: TEAL, width: 1.2 } });
    s.addImage({ data: ICON.scissors, x: 7.4, y: 5.65, w: 0.45, h: 0.45 });
    s.addText([{ text: "Tách câu bằng LLM  ", options: { bold: true, color: TEAL } }, { text: "cải thiện Micro-Recall trên mọi cấu hình (+1–2 điểm %); T5 hưởng lợi nhiều hơn (BiToMe+CDM +2.24).", options: { color: INK } }], { x: 7.95, y: 5.57, w: PW - 7.15 - MX - 1.0, h: 0.85, margin: 0, fontFace: FS, fontSize: 11, lineSpacingMultiple: 1.12, valign: "middle" });
    s.addNotes("TASD: SLM-CDM tốt trên BERT, BiToMe-CDM tốt trên T5 (71.18%). Resize > compact +4.29pp. LLM tách câu tăng Recall 1-2pp.");
  }

  // ---------------- SLIDE 20 : Kết luận & hướng phát triển ----------------
  {
    const s = pres.addSlide();
    s.background = { color: NAVY };
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 0.42, w: 0.62, h: 0.62, rectRadius: 0.12, fill: { color: MINT }, line: { type: "none" } });
    s.addImage({ data: ICON.flag, x: MX + 0.14, y: 0.56, w: 0.34, h: 0.34 });
    s.addText("KẾT LUẬN & HƯỚNG PHÁT TRIỂN", { x: MX + 0.86, y: 0.4, w: 9, h: 0.28, margin: 0, fontFace: FS, fontSize: 11.5, bold: true, color: MINT, charSpacing: 2, valign: "bottom" });
    s.addText("ToMe khả thi cho ABSA khi kết hợp bảo vệ ngữ cảnh cục bộ", { x: MX + 0.86, y: 0.66, w: 11.5, h: 0.56, margin: 0, fontFace: FH, fontSize: 24, bold: true, color: WHITE, valign: "middle" });
    s.addText("20", { x: PW - 1.15, y: 0.42, w: 0.6, h: 0.5, margin: 0, fontFace: FS, fontSize: 15, bold: true, color: "3A5876", align: "right", valign: "middle" });

    // conclusions (left)
    s.addText("KẾT LUẬN CHÍNH", { x: MX, y: 1.5, w: 6, h: 0.3, margin: 0, fontFace: FS, fontSize: 11.5, bold: true, color: MINT, charSpacing: 1.5 });
    const concl = [
      "ToMe áp dụng được cho ABSA mà không suy giảm đáng kể chất lượng; một số cấu hình còn cải thiện Micro/Macro-F1.",
      "LCF đóng vai trò then chốt: bảo vệ vùng khía cạnh giúp tránh hợp nhất nhầm token quan trọng — lợi ích chỉ rõ khi ToMe & LCF kết hợp.",
      "BiToMe+CDM cân bằng tốt nhất (đỉnh T5, TASD 71.18%); SCM+CDM cho Macro-F1 cao nhất trên BERT (71.98%).",
      "Tách câu bằng LLM cải thiện Recall; sai số ATE lan truyền cho thấy chất lượng trích xuất là yếu tố quyết định.",
    ];
    let cy = 1.85;
    concl.forEach((c, i) => {
      s.addShape(pres.shapes.OVAL, { x: MX, y: cy, w: 0.42, h: 0.42, fill: { color: TEAL }, line: { type: "none" } });
      s.addText(String(i + 1), { x: MX, y: cy, w: 0.42, h: 0.42, margin: 0, fontFace: FH, fontSize: 14, bold: true, color: WHITE, align: "center", valign: "middle" });
      s.addText(c, { x: MX + 0.6, y: cy - 0.08, w: 6.0, h: 1.05, margin: 0, fontFace: FS, fontSize: 12, color: "D5E1EC", lineSpacingMultiple: 1.12, valign: "top" });
      cy += 1.12;
    });

    // right: limitations + future
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.35, y: 1.65, w: PW - 7.35 - MX, h: 1.95, rectRadius: 0.1, fill: { color: NAVY2 }, line: { color: CORAL, width: 1.2 } });
    s.addImage({ data: ICON.warningW, x: 7.6, y: 1.9, w: 0.42, h: 0.42 });
    s.addText("Hạn chế", { x: 8.12, y: 1.9, w: 4, h: 0.42, margin: 0, fontFace: FS, fontSize: 14, bold: true, color: WHITE, valign: "middle" });
    s.addText([
      { text: "Một bộ dữ liệu, một miền (khách sạn, tiếng Anh), mất cân bằng nặng.", options: { bullet: { indent: 12 }, breakLine: true } },
      { text: "ToMe áp dụng sau backbone → chưa giảm chi phí self-attention; chưa đo FLOPs / thời gian suy luận.", options: { bullet: { indent: 12 }, breakLine: true } },
      { text: "Mỗi cấu hình chỉ 1 seed → độ tin cậy thống kê hạn chế.", options: { bullet: { indent: 12 } } },
    ], { x: 7.6, y: 2.4, w: PW - 7.35 - MX - 0.5, h: 1.15, margin: 0, fontFace: FS, fontSize: 10.5, color: "C9D6E2", lineSpacingMultiple: 1.08, paraSpaceAfter: 5, valign: "top" });

    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.35, y: 3.8, w: PW - 7.35 - MX, h: 2.7, rectRadius: 0.1, fill: { color: NAVY2 }, line: { color: MINT, width: 1.2 } });
    s.addImage({ data: ICON.rocketW, x: 7.6, y: 4.05, w: 0.42, h: 0.42 });
    s.addText("Hướng phát triển", { x: 8.12, y: 4.05, w: 4, h: 0.42, margin: 0, fontFace: FS, fontSize: 14, bold: true, color: WHITE, valign: "middle" });
    s.addText([
      { text: "Dynamic Token Merging — tự chọn chiến lược & số token theo câu/tầng.", options: { bullet: { indent: 12 }, breakLine: true } },
      { text: "Áp dụng ToMe từ các tầng đầu Transformer + khai thác compact để đo tăng tốc thực sự.", options: { bullet: { indent: 12 }, breakLine: true } },
      { text: "Backbone hiện đại (DeBERTa-v3, ModernBERT, LLMs); mô hình sinh end-to-end giảm lan truyền lỗi.", options: { bullet: { indent: 12 }, breakLine: true } },
      { text: "Mở rộng đa miền, đa ngôn ngữ & đa phương thức (Multimodal ABSA).", options: { bullet: { indent: 12 } } },
    ], { x: 7.6, y: 4.55, w: PW - 7.35 - MX - 0.5, h: 1.9, margin: 0, fontFace: FS, fontSize: 10.5, color: "C9D6E2", lineSpacingMultiple: 1.1, paraSpaceAfter: 6, valign: "top" });

    // footer thanks
    s.addText("Cảm ơn Thầy Cô và các bạn đã lắng nghe.", { x: MX, y: 6.75, w: 8, h: 0.35, margin: 0, fontFace: FH, fontSize: 13, italic: true, color: MINT });
    s.addNotes("Kết luận: ToMe khả thi cho ABSA khi kết hợp LCF. BiToMe+CDM cân bằng nhất, SCM+CDM Macro-F1 cao nhất. Hạn chế + hướng phát triển: Dynamic ToMe, backbone hiện đại, đa ngôn ngữ/đa phương thức.");
  }

  await pres.writeFile({ fileName: "/home/claude/LCTA_KhoaLuan.pptx" });
  console.log("WROTE deck");
}

main().catch((e) => { console.error(e); process.exit(1); });