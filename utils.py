from fpdf import FPDF


def get_triples(path):
    print("Reading ", path)
    with open(path) as ef:
        nE = int(ef.readline())
        triples = []
        for line in ef:
            line = line.strip()
            h,t,r = line.split(" ")
            h,t,r = int(h), int(t), int(r)
            triples.append((h,r,t))

    return triples

def get_list(path):
    print("Reading ", path)
    with open(path) as ef:
        nE = int(ef.readline())
        print(f"Num Items: {nE}")
        entities = {}
        for line in ef:
            line = line.strip()
            name,id = line.split("\t")
            entities[id] = name

    return nE, entities


def combine2pdf(imagelist, f="plots/combined.pdf"):
    pdf = FPDF()
    # imagelist is the list with all image filenames
    for images in imagelist:
        pdf.add_page()
        y = 10
        for image in images:
            pdf.image(image, 10, y, 65, 60)
            y += 63
    pdf.output(f, "F")
