import csv

def Get_Labels(Column_Contents):
    with open('CODE_TB.csv', 'r') as csvfile:
        Raw = list(csv.reader(csvfile))
    Labels = []
    for row in Column_Contents:
        a = row[0]
        for item in row[1]:
            b = item
            d = ''
            for c in Raw:
                if c[1].lower() == a.lower() and c[2]==item:
                    d = c[3]
            Labels.append([a, b, d])
    return Labels
