# Stores prompt templates for summarization and querying
# Sistem mesajı: GPT'ye genel rolünü tanımlar
SYSTEM_MESSAGE = "Sen, kullanıcıya akademik araştırma süreçlerinde yardımcı olan, makaleleri özetleyen ve analiz eden bir yapay zekâ agentsın."

# Özetleme şablonu
SUMMARY_PROMPT_TEMPLATE = """
Sen bir akademik araştırma asistanısın.

Aşağıda başlığı ve özeti verilen bir makaleyi, sade ve kısa bir dille (3-4 cümle ile) özetle. 
Teknik terimlerde açıklık sağla, tekrar etme.

Başlık: {title}

Özet:
{abstract}

Kısa Özet:
"""

# Referans formatı üretimi için (isteğe bağlı kullanılacak)
CITATION_PROMPT_TEMPLATE = """
Aşağıda başlığı, yazarları, yayın yılı ve URL'si verilen bir akademik makale yer almaktadır.

Bu makale için APA ve BibTeX referans formatlarını oluştur.

Başlık: {title}
Yazarlar: {authors}
Yıl: {year}
URL: {url}

APA:
...

BibTeX:
...
"""
