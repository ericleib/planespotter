# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy

class PlaneItem(scrapy.Item):
    immat = scrapy.Field()
    cn = scrapy.Field()
    airline = scrapy.Field()
    manuf = scrapy.Field()
    model = scrapy.Field()

    image_urls = scrapy.Field()
    images = scrapy.Field()
