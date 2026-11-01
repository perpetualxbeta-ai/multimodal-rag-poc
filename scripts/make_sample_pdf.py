"""Generate samples/halcyon_fy2025_report.pdf, a fictional annual report for testing the extractor.

Usage: python scripts/make_sample_pdf.py samples/halcyon_fy2025_report.pdf  (needs reportlab, matplotlib)
"""

import io
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = sys.argv[1]
COMPANY = "Halcyon Grid Energy"

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=14, spaceAfter=8)
caption = ParagraphStyle("caption", parent=body, fontSize=9, textColor=colors.HexColor("#444444"), alignment=TA_CENTER)
h1 = styles["Heading1"]
h2 = styles["Heading2"]


def png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# --- raster charts (matplotlib PNGs) ---------------------------------------------------------
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
gen_2024 = [118, 124, 141, 152, 166, 171, 175, 172, 158, 143, 127, 119]
gen_2025 = [131, 139, 158, 174, 191, 199, 204, 201, 183, 165, 146, 137]
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(months, gen_2024, marker="o", label="FY2024", color="#9aa5b1")
ax.plot(months, gen_2025, marker="o", label="FY2025", color="#e07a1f", linewidth=2.2)
ax.set_ylabel("Net generation (GWh)")
ax.set_title("Monthly net generation")
ax.grid(alpha=0.3)
ax.legend()
gen_chart = png(fig)

mix = {"Solar": 58, "Wind": 27, "Battery discharge": 11, "Hydro": 4}
fig, ax = plt.subplots(figsize=(4.2, 3.4))
ax.pie(mix.values(), labels=mix.keys(), autopct="%1.0f%%", startangle=90,
       colors=["#f2b134", "#4f9dde", "#6cc070", "#2a6f97"])
ax.set_title("FY2025 generation mix")
mix_chart = png(fig)

# --- vector bar chart (reportlab graphics, stays as vector drawing in the PDF) ----------------
regions = ["Northern", "Coastal", "Central", "Southern"]
cap_2024 = [360, 280, 240, 150]
cap_2025 = [420, 310, 265, 185]
bar = Drawing(15 * cm, 7 * cm)
chart = VerticalBarChart()
chart.x, chart.y, chart.width, chart.height = 40, 30, 15 * cm - 60, 7 * cm - 50
chart.data = [cap_2024, cap_2025]
chart.categoryAxis.categoryNames = regions
chart.valueAxis.valueMin, chart.valueAxis.valueMax, chart.valueAxis.valueStep = 0, 450, 50
chart.bars[0].fillColor = colors.HexColor("#9aa5b1")
chart.bars[1].fillColor = colors.HexColor("#e07a1f")
chart.barSpacing, chart.groupSpacing = 2, 14
bar.add(chart)
bar.add(Rect(15 * cm - 120, 7 * cm - 18, 8, 8, fillColor=colors.HexColor("#9aa5b1"), strokeColor=None))
bar.add(String(15 * cm - 108, 7 * cm - 17, "2024", fontSize=8))
bar.add(Rect(15 * cm - 75, 7 * cm - 18, 8, 8, fillColor=colors.HexColor("#e07a1f"), strokeColor=None))
bar.add(String(15 * cm - 63, 7 * cm - 17, "2025", fontSize=8))

# --- vector diagram: energy flow -------------------------------------------------------------
flow = Drawing(16 * cm, 4 * cm)
steps = [("Solar & wind\nfarms", "#f2b134"), ("Inverters &\nsubstations", "#4f9dde"),
         ("Battery\nstorage", "#6cc070"), ("Regional\ngrid", "#2a6f97")]
box_w, box_h, gap = 3.1 * cm, 2.0 * cm, 0.9 * cm
for i, (label, colour) in enumerate(steps):
    x = 0.2 * cm + i * (box_w + gap)
    flow.add(Rect(x, 1 * cm, box_w, box_h, rx=6, ry=6, fillColor=colors.HexColor(colour), strokeColor=colors.black))
    for j, line in enumerate(label.split("\n")):
        flow.add(String(x + box_w / 2, 1 * cm + box_h / 2 + 3 - j * 11, line, fontSize=9, textAnchor="middle"))
    if i < len(steps) - 1:
        ax0, ax1, ay = x + box_w + 3, x + box_w + gap - 3, 1 * cm + box_h / 2
        flow.add(Line(ax0, ay, ax1 - 6, ay, strokeWidth=1.5))
        flow.add(Polygon([ax1, ay, ax1 - 7, ay + 4, ax1 - 7, ay - 4], fillColor=colors.black, strokeColor=None))

# --- tables -----------------------------------------------------------------------------------
rev = [42.1, 47.8, 55.3, 61.0]
cost = [30.5, 33.9, 37.6, 40.2]
fin_rows = [["Quarter", "Revenue ($M)", "Operating costs ($M)", "Operating profit ($M)", "Margin"]]
for q, r, c in zip(["Q1", "Q2", "Q3", "Q4"], rev, cost, strict=True):
    fin_rows.append([q, f"{r:.1f}", f"{c:.1f}", f"{r - c:.1f}", f"{(r - c) / r:.1%}"])
tr, tc = sum(rev), sum(cost)
fin_rows.append(["FY2025", f"{tr:.1f}", f"{tc:.1f}", f"{tr - tc:.1f}", f"{(tr - tc) / tr:.1%}"])

site_rows = [
    ["Site", "Region", "Type", "Capacity (MW)", "Commissioned"],
    ["Redfern Ridge", "Northern", "Solar", "180", "2019"],
    ["Kestrel Bay", "Coastal", "Offshore wind", "310", "2021"],
    ["Ambervale", "Central", "Solar + storage", "145", "2023"],
    ["Pinecrest", "Northern", "Wind", "240", "2018"],
    ["Saltmarsh", "Southern", "Solar", "185", "2025"],
    ["Linden Falls", "Central", "Hydro", "120", "2012"],
]


def grid_table(rows: list[list[str]], widths: list[float]) -> Table:
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9edf1")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


# --- small header logo (should be ignored by the extractor) -----------------------------------
fig, ax = plt.subplots(figsize=(0.5, 0.5))
ax.add_patch(plt.Circle((0.5, 0.5), 0.45, color="#e07a1f"))
ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
logo = ImageReader(png(fig))


def on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.drawImage(logo, A4[0] - 2.4 * cm, A4[1] - 1.6 * cm, width=0.8 * cm, height=0.8 * cm, mask="auto")
    canvas.setFont("Helvetica", 8)
    canvas.drawString(2 * cm, A4[1] - 1.3 * cm, f"{COMPANY} - FY2025 Annual Operations Report (fictional sample)")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


P = lambda text: Paragraph(text, body)  # noqa: E731
story = [
    Paragraph(f"{COMPANY}<br/>FY2025 Annual Operations Report", styles["Title"]),
    Paragraph("This is a fictional document generated for testing a multimodal RAG pipeline. "
              "All companies, sites and figures are invented.", caption),
    Spacer(1, 12),
    Paragraph("1. Executive summary", h1),
    P(f"{COMPANY} delivered its strongest year to date in FY2025. Net generation rose to 2,028 GWh, "
      "up 14.8% on FY2024, driven by the commissioning of the Saltmarsh solar farm in the Southern "
      "region and higher availability at the Kestrel Bay offshore wind site."),
    P("Revenue grew in every quarter, reaching $206.2 million for the year, while operating margin "
      "expanded from 27.6% in Q1 to 34.1% in Q4. The improvement came mainly from lower curtailment "
      "and the shift of evening demand onto battery storage, which reduced spot-market purchases."),
    P("Installed capacity reached 1,180 MW across four regions. Safety performance remained strong, "
      "with a lost-time injury frequency rate of 0.4 per million hours worked, the lowest in the "
      "company's history."),
    Image(gen_chart, width=15 * cm, height=7.3 * cm),
    Paragraph("Figure 1: Monthly net generation in GWh, FY2024 versus FY2025. Output peaks in July "
              "at 204 GWh and is lowest in January.", caption),
    PageBreak(),
    Paragraph("2. Financial results", h1),
    P("Quarterly results are summarised in Table 1. Revenue increased from $42.1 million in Q1 to "
      "$61.0 million in Q4. Operating costs grew more slowly than revenue because maintenance "
      "contracts were renegotiated in the second quarter."),
    grid_table(fin_rows, [2.4 * cm, 2.8 * cm, 3.6 * cm, 3.7 * cm, 2.2 * cm]),
    Paragraph("Table 1: Quarterly financial results for FY2025.", caption),
    Spacer(1, 10),
    P("Q3 was the strongest quarter for profit growth. Operating profit rose to $17.7 million as "
      "summer output peaked and the Ambervale battery system shifted 310 MWh per day of solar "
      "generation into the evening peak, when wholesale prices are highest."),
    P("Capital expenditure for the year was $88.4 million, of which $61.2 million went to the "
      "Saltmarsh build-out and the remainder to grid connection upgrades in the Coastal region. "
      "Net debt at year end stood at $312 million, or 2.1 times EBITDA."),
    Paragraph("2.1 Cost drivers", h2),
    P("The largest cost categories were operations and maintenance (41% of operating costs), "
      "network charges (23%), staff (19%) and insurance (9%). Insurance premiums rose 14% after "
      "a regional storm season, partly offset by lower turbine gearbox replacements."),
    PageBreak(),
    Paragraph("3. Generation portfolio", h1),
    P("Figure 2 compares installed capacity by region. The Northern region remains the largest, "
      "with 420 MW, while the Southern region grew fastest, up 23% following the Saltmarsh "
      "commissioning."),
    bar,
    Paragraph("Figure 2: Installed capacity by region in MW, 2024 versus 2025.", caption),
    Spacer(1, 10),
    P("Table 2 lists the six operating sites. Kestrel Bay is the largest single site at 310 MW; "
      "Linden Falls, a run-of-river hydro scheme, is the oldest asset in the portfolio."),
    grid_table(site_rows, [3.4 * cm, 2.6 * cm, 3.4 * cm, 2.8 * cm, 2.8 * cm]),
    Paragraph("Table 2: Operating sites as of 31 December 2025.", caption),
    PageBreak(),
    Paragraph("4. Energy mix and system design", h1),
    P("Solar was the largest source of generation in FY2025 at 58%, followed by wind at 27%. "
      "Battery discharge contributed 11% of delivered energy, up from 6% the previous year, "
      "and hydro provided the remaining 4%."),
    Image(mix_chart, width=9 * cm, height=7.3 * cm),
    Paragraph("Figure 3: FY2025 generation mix by source.", caption),
    Spacer(1, 10),
    P("Energy flows from the generation sites through inverters and substations, where it is "
      "either stored in battery systems or exported directly to the regional grid. Battery "
      "dispatch is scheduled day-ahead using price and weather forecasts."),
    flow,
    Paragraph("Figure 4: Simplified energy flow from generation to the regional grid.", caption),
    PageBreak(),
    Paragraph("5. Outlook and risks", h1),
    P("For FY2026 the company expects net generation of 2,150 to 2,250 GWh and capital expenditure "
      "of around $95 million, mainly for a 100 MW battery extension at Ambervale and repowering "
      "of the oldest Pinecrest turbines."),
    P("The main risks to this outlook are:"),
    ListFlowable([
        ListItem(P("Grid curtailment in the Coastal region if transmission upgrades are delayed beyond Q3 2026.")),
        ListItem(P("Lower solar yield in a La Nina year, which historically reduces output by 4-6%.")),
        ListItem(P("Battery cell supply constraints affecting the Ambervale extension timeline.")),
        ListItem(P("Changes to the regional capacity market rules expected to be finalised in mid-2026.")),
    ], bulletType="bullet"),
    P("The board has approved a dividend of 8.5 cents per share, a payout ratio of 45% of net "
      "profit, and reaffirmed the target of reaching 1,500 MW of installed capacity by 2028."),
]

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2.2 * cm,
                        bottomMargin=2 * cm, title=f"{COMPANY} FY2025 Annual Operations Report (sample)")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT)
