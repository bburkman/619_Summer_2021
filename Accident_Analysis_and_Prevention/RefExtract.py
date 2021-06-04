from refextract import extract_references_from_file

references = extract_references_from_file('./Already_Read/A-deep-learning-based-traffic-crash-severity-pre_2021_Accident-Analysis---Pr.pdf')
for ref in references:
    print (ref)
    print ()
