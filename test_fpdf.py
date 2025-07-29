from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Hello, this is a test PDF.", ln=True, align="C")
pdf.output("test_output.pdf")