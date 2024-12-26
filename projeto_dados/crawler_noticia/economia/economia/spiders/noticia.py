import scrapy

class NoticiasSpider(scrapy.Spider):
    name = 'noticia'
    custom_settings = {'FEED_EXPORT_ENCODING': 'utf-8'} 
    start_urls = ['https://g1.globo.com/economia/']

    def parse(self, response):
        noticias = response.xpath('//div[@class="feed-post-body"]')
        noticia_por_nome = {}

        for noticia in noticias:
            cidade = noticia.xpath('.//span/text()').get()
            previsao = noticia.xpath('.//p/text()').get()

            if cidade in noticia_por_nome:
                noticia_por_nome[cidade].append(previsao)
            else:
                noticia_por_nome[cidade] = [previsao]

        yield noticia_por_nome
        





