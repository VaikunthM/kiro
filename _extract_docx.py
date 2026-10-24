import zipfile, re
for f in ['robotics_idea.docx', 'Task Plan.docx']:
    print('=' * 20, f, '=' * 20)
    try:
        with zipfile.ZipFile(f) as z:
            xml = z.read('word/document.xml').decode('utf-8', 'ignore')
        for p in re.split(r'</w:p>', xml):
            line = ''.join(re.findall(r'<w:t[^>]*>(.*?)</w:t>', p))
            if line.strip():
                print(line)
    except Exception as e:
        print('ERR', e)
