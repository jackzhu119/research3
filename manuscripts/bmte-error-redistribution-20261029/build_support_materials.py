"""Prepare declared author metadata and the journal's separate official form."""
from pathlib import Path
import csv,json
from docx import Document
from project_config import *

P=Path(__file__).resolve().parent

def main():
    author=json.loads((P/'submission/author_information.json').read_text())
    author.update(affiliation=ADDRESS,target_journal=JOURNAL,current_manuscript_title=TITLE,
        manuscript_stage='Integrated author-review version; not submitted',
        previous_journal_submission_scope='Current integrated version not submitted; earlier related conventional-decoder version recorded as rejected by PLOS ONE',
        current_submission_performed=False)
    author.pop('previous_journal_submission',None)
    author['earlier_public_manuscripts']=[
        'https://github.com/jackzhu119/few-shot-mi-eeg/blob/da2a3acee24746981bbcdd779ab88132da6e24c4/manuscript/bmt_component_revision_20261029/MAIN.md',
        'https://github.com/jackzhu119/research3/tree/e5d555ebc6b8f89fc5163cf446304dea9e35a0dd/manuscripts/brain-sciences-erfc-20261029',
        'https://github.com/jackzhu119/cross-subject-mi-eeg/blob/c4ea1f4ee1d078bcb015922f8092f05e263f2399/research_runs/PAPER1_IJBET_20261029/README_zh.md']
    (P/'submission/author_information.json').write_text(json.dumps(author,ensure_ascii=False,indent=2)+'\n')
    with (P/'submission/authors_for_submission.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['order','given_name','family_name','email','orcid','corresponding','affiliation']);writer.writeheader()
        for i,a in enumerate(author['authors'],1):writer.writerow({'order':i,'given_name':a['given'],'family_name':a['family'],'email':a['email'],'orcid':a.get('orcid',''),'corresponding':bool(a.get('corresponding',False)),'affiliation':ADDRESS})
    declarations={'Acknowledgments':ACK,'Research ethics':ETHICS,'Informed consent':CONSENT,
        'Author contributions (for double- anonymized journals: please use initials)':CONTRIBUTIONS,
        'Use of Large Language Models, AI and Machine Learning Tools':AI,'Conflict of interest':COI,'Research funding':FUNDING,'Data availability':DATA}
    template=P/'research/ethical_declarations_template.docx'
    if not template.exists():template=P/'submission/Ethical_and_Legal_Declarations.docx'
    doc=Document(template)
    filled=0
    for row in doc.tables[0].rows[1:]:
        k=row.cells[0].text.strip()
        if k in declarations:row.cells[1].text=declarations[k];filled+=1
    assert filled==8
    doc.core_properties.author='Ziyuan Zhu; Sihao Liu; Haoyang Yu';doc.core_properties.title=TITLE+' — ethical and legal declarations'
    doc.save(P/'submission/Ethical_and_Legal_Declarations.docx')
    (P/'submission/Ethical_and_Legal_Declarations.md').write_text('# Ethical and legal declarations\n\n'+'\n\n'.join('## '+k+'\n\n'+v for k,v in declarations.items())+'\n')
    header=f'''# {TITLE}

**Ziyuan Zhu\\*, Sihao Liu and Haoyang Yu**

{ADDRESS}

\\* Corresponding author: Ziyuan Zhu; {ADDRESS}; e-mail: zzy2630816871@gmail.com

Sihao Liu: liu3100853087@gmail.com; Haoyang Yu: jiaguca3329@163.com  
Ziyuan Zhu ORCID: <https://orcid.org/0009-0005-1153-4926>

**Running head:** {RUNNING_HEAD}
'''
    (P/'submission/Title_page.md').write_text(header)
    print(json.dumps({'declaration_fields_filled':filled,'authors':3,'postcode':'401331','submitted':False}))

if __name__=='__main__':main()
