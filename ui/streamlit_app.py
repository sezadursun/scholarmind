from pathlib import Path
import re

src_path = Path("/mnt/data/Pasted markdown(1).md")
raw = src_path.read_text(encoding="utf-8")

# Kullanıcının dosyasında yanlışlıkla kalan wrapper'ı temizle:
# from pathlib import Path
# code = r''' ... '''
# out = Path(...)
# out.write_text(...)
match = re.search(r"code\s*=\s*r'''(.*)'''\s*out\s*=", raw, flags=re.DOTALL)

if match:
    cleaned = match.group(1).strip() + "\n"
else:
    # Eğer wrapper yoksa dosyayı olduğu gibi kullan
    cleaned = raw.strip() + "\n"

out_path = Path("/mnt/data/streamlit_app.py")
out_path.write_text(cleaned, encoding="utf-8")

out_path.as_posix()
