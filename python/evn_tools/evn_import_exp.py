import re
import pathlib
import pyvo
import gzip
from urllib.request import urlopen, urlretrieve
from urllib.parse import urlparse

service = pyvo.dal.TAPService('http://evn-vo.jive.eu/tap')

def evn_import_exp(expname):
    query = service.search(f"SELECT * FROM ivoa.obscore WHERE obs_id = '{expname}'")
    datalink = query[0].getdatalink()
    urls = [f.access_url for f in datalink]
    if type(urls[0]) == bytes:
        urls = [x.decode() for x in urls]

    for url in urls:
        filename = pathlib.Path(urlparse(url).path).name
        urlretrieve(url, filename)

    # Currently the calibration files are not in the VO, as work around try to get these from 
    # the archive using some heuristics
    with urlopen('http://archive.jive.nl/exp/') as response:
        lines = response.read().decode('utf-8').split('\n')
    exp = expname.upper()
    r = re.compile(r'href=\"(\w+)_(\d+)')
    dirlines = [i for i in lines if 'alt="[DIR]"' in i]
    dirname = ""
    for line in dirlines:
        result = r.findall(line)
        if (len(result) == 1) and (len(result[0]) == 2) and (result[0][0] == exp):
            dirname = '_'.join(result[0])
            break
    if dirname == "":
        print("WARNING: No calibration tables loaded from archive")
        return
    url = f'http://archive.jive.nl/exp/{dirname}/pipe'
    with urlopen(url) as response:
        lines = response.read().decode('utf-8').split('\n')
    r = re.compile(r'href=\"([-+\w\.]*)\"')
    for line in lines:
        result = r.findall(line)
        if len(result) == 1:
            lowername = result[0].lower()
            if lowername.endswith('.antab.gz') or lowername.endswith('.uvflg'):
                urlretrieve(url + '/' + result[0], result[0])
                # decompress antab
                if result[0].endswith('.gz'):
                    with gzip.open(result[0], 'rb') as f:
                        antab = f.read()
                    antabname = result[0][:-3]
                    with open(antabname, 'wb') as f:
                        f.write(antab)
