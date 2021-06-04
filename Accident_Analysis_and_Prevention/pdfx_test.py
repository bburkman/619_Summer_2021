import pdfx
pdf = pdfx.PDFx("./Already_Read/Key-risk-indicators-for-accident-assessment-condition_2018_Accident-Analysis.pdf")
metadata = pdf.get_metadata()
references_list = pdf.get_references()
references_dict = pdf.get_references_as_dict()
stuff = pdf.get_text()
print (stuff)
