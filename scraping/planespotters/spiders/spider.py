import scrapy
import re
import urlparse
import urllib
import unicodedata

#from scrapy.contrib.spiders import CrawlSpider, Rule
#from scrapy.contrib.linkextractors import LinkExtractor

from planespotters.items import PlaneItem

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

class PlaneSpottersSpider(scrapy.Spider):
    name = 'ps'
    allowed_domains = ["planespotters.net"]
    start_urls = [
        "http://www.planespotters.net/Aviation_Photos/search.php"
    ]
    #rules = (
    #    Rule(LinkExtractor(allow=('search.php'), restrict_xpaths="//*[@id='PaginationNexttop']"), callback='parse_item'),
    #)

    known_fields = {
        "immat": "search for registration:",
        "cn": "search for construction number:",
        "manuf": "search for manufacturer:",
        "model": "search for aircraft type:",
        "airline": "search for airline:"
    }

    def parse(self, response):
        print "Parsing a new page: " + response.url
        for sel in response.css('.PhotoContainer .PhotoDetail').xpath('tr[2]'):
            print "New photo!"            
            item = PlaneItem();
            
            # Get fields 
            for field in sel.xpath('td[2]/a/@title').extract():
                field = strip_accents(field.lower().strip())
                for code,pattern in self.known_fields.iteritems():
                    if field.startswith(pattern):
                        item[code] = field[len(pattern):].strip()
                        print "Found: "+code+" = "+item[code]
                        break
            
            # Get image
            paths = sel.xpath('td[1]/a/img/@src').extract()
            item['image_urls'] = map(self.add_http, paths)

            yield item
 
        next_page = response.xpath("//*[@id='PaginationNexttop']").extract()[0]
        url = re.search('href=([\"\'])(.*?)\\1', next_page, re.IGNORECASE).group(2)
        yield scrapy.Request(urlparse.urljoin(response.url, url), callback=self.parse)

    def add_http(self,url):
        if(not url.lower().startswith('http:')):
            return 'http:'+url
        return url

  
  
class AirlinersSpider(scrapy.Spider):
    name = 'airliners'
    allowed_domains = ["airliners.net"]
    start_urls = [
        "http://www.airliners.net/search/photo.search?specialsearch=airliners&sort_order=photo_id+desc&page_limit=500&page=1236&sid=80937f8fd75a3a226cff0c29b79b3f21"
    ]
    #rules = (
    #    Rule(LinkExtractor(allow=('search.php'), restrict_xpaths="//*[@id='PaginationNexttop']"), callback='parse_item'),
    #)

    known_fields = {
        "immat": "regsearch=",
        "cn": "cnsearch=",
        #"manuf": "search for manufacturer:",
        "model": "aircraft_genericsearch=",
        "airline": "airlinesearch=="
    }

    def parse(self, response):
        print "Parsing a new page: " + response.url
        for sel in response.css('.p402_premium').xpath('table/tr[1]/td[1]/table'):
            print "New photo!"            
            item = PlaneItem();
            
            # Get fields 
            for field in sel.xpath('tr//a/@href').extract():
                #print "Raw field: "+field
                field = strip_accents(re.sub("\(.*?\)", "", urllib.unquote(field)).strip().lower()) # strip, decode url, remove everything in brackets, remove accents, to lower case
                for code,pattern in self.known_fields.iteritems():
                    if pattern in field:
                        res = re.search(pattern+'(.*?)&distinct', field)
                        if res:
                            data = res.group(1).encode('utf8')
                        else:
                            data = "unknown"
                        if data != "-":
                            if code == "model":
                                print "Data (model) : " + data
                                res = re.match('([^0-9]+) +(.*)', data)
                                if res:
                                    item['manuf'] = res.group(1)
                                    item['model'] = re.sub(" ?[/ ].*","",res.group(2))
                                    print "Found: model = "+item['model']
                                    print "Found: manuf = "+item['manuf']
                                else:
                                    item['model'] = data
                                    print "Found: model = "+item['model']+" (but manufacturer missing!)"
                            else:
                                item[code] = data
                                print "Found: "+code+" = "+item[code]
                        break
            
            # Get image
            paths = [sel.xpath('tr//img/@src').extract()[0]]
            item['image_urls'] = map(self.add_http, paths)

            yield item
 
        url = response.xpath("//table")[0].xpath("tr[1]/td[last()]/a/@href").extract()[0]
        yield scrapy.Request(urlparse.urljoin(response.url, url), callback=self.parse)

    def add_http(self,url):
        if(not url.lower().startswith('http:')):
            return 'http:'+url
        return url




 
  
  
  
  
  
 
  
  
  
  
  
  
  
 
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
 
  
  
  
  
  
 
  
  
  
  
 


