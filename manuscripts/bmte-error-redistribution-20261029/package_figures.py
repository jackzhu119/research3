"""Package the eight quantitative figures, editable vectors and plotting inputs."""
from pathlib import Path
import zipfile,json,hashlib

P=Path(__file__).resolve().parent

def main():
    folder=P/'figures'
    names=[str(f.relative_to(P))for f in folder.rglob('*')if f.is_file()and '__pycache__'not in f.parts and f.suffix!='.pyc']
    names+=['requirements-analysis.txt']
    assert len(list(folder.glob('Fig*.tif')))==8
    out=P/'Scientific_Figures_All_8.zip'
    manifest={'scientific_figures':8,'main_figures':4,'supplementary_figures':4,'renderer':'Matplotlib','raster_dpi':600,'raster_colour_mode':'RGB',
              'formats':['editable SVG','vector PDF','PNG','LZW-compressed TIFF'],'files':[{'path':n,'bytes':(P/n).stat().st_size,'sha256':hashlib.sha256((P/n).read_bytes()).hexdigest()}for n in sorted(names)]}
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
        for n in sorted(names):z.write(P/n,n)
        z.writestr('FIGURE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(out)as z:
        assert z.testzip()is None
        for r in manifest['files']:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    receipt={'file':out.name,'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'scientific_figures':8,'all_member_hashes_verified':True}
    (P/'FIGURE_PACKAGE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
