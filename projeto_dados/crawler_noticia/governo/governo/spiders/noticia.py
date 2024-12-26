import scrapy

class G1Spider(scrapy.Spider):
    name = 'noticia'
    custom_settings = {'FEED_EXPORT_ENCODING': 'utf-8'} 
    start_urls = [
        'https://g1.globo.com/',
        'https://g1.globo.com/brasil/',
        'https://g1.globo.com/mundo/',
        # adicione mais URLs para outras seções de notícias
    ]

    def parse(self, response):
        for link in response.css('.feed-post-link::attr(href)').getall():
            yield response.follow(link, self.parse_news)

    def parse_news(self, response):
        # Extrair informações relevantes das notícias
        title = response.css('.content-head__title::text').get()
        summary = response.css('.content-head__subtitle::text').get()
        body = response.css('.content-text__container *::text').getall()

        # Tratar casos onde os elementos podem ser nulos
        title = title if title is not None else ''
        summary = summary if summary is not None else ''
        body = ''.join(body).strip() if body is not None else ''

        # Criar o dicionário aninhado
        item = {
            'title': {'text': title},
            'summary': {'text': summary},
            'body': {'text': body}
        }
        yield item



