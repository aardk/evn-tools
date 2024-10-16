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
    if (len(urls) > 0) and (type(urls[0]) == bytes):
        urls = [x.decode() for x in urls]

    for url in urls:
        filename = pathlib.Path(urlparse(url).path).name
        urlretrieve(url, filename)
        if filename.lower().endswith('.antab.gz'):
            with gzip.open(filename, 'rb') as f:
                antab = f.read()
            antabname = filename[:-3]
            with open(antabname, 'wb') as f:
                f.write(antab)
