import sys,asyncio
from playwright.async_api import async_playwright
async def main(svg,png,scale):
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
        pg=await b.new_page(viewport={'width':894,'height':1080},device_scale_factor=float(scale))
        await pg.goto('file://'+svg); await pg.screenshot(path=png,full_page=False); await b.close()
asyncio.run(main(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else 1.5))
