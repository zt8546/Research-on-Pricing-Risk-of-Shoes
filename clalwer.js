(async function() {
    const minSize = 8192; 
    let downloadedUrls = new Set();
    let fileIndex = 1;
    const prefix = "Converse_";
    const scrollInterval = setInterval(() => {
        window.scrollBy(0, window.innerHeight); 
    }, 2500);
    const getNewUrls = () => {
        const imgs = document.querySelectorAll('img');
        const currentBatch = Array.from(imgs)
            .map(img => {
                return img.getAttribute('data-src') || 
                       img.getAttribute('data-original') || 
                       img.getAttribute('original-src') || 
                       img.currentSrc || 
                       img.src;
            })
            .filter(src => 
                src && 
                src.startsWith('http') && 
                !src.toLowerCase().includes('favicon') && 
                !src.includes('placeholder') && 
                !downloadedUrls.has(src)
            );
        return [...new Set(currentBatch)];
    };
    try {
        while (true) {
            const targetUrls = getNewUrls();
            if (targetUrls.length === 0) {
                await new Promise(r => setTimeout(r, 2000));
                continue;
            }
            for (let url of targetUrls) {
                downloadedUrls.add(url);
                const fileName = `${prefix}${fileIndex}.jpg`;
                try {
                    const res = await fetch(url);
                    const blob = await res.blob();
                    if (blob.size < minSize) continue;
                    const blobUrl = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = blobUrl;
                    a.download = fileName;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(blobUrl);
                    fileIndex++; 
                    await new Promise(r => setTimeout(r, 800));
                } catch (e) {
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = fileName;
                    a.target = '_blank';
                    a.click();
                    fileIndex++; 
                    await new Promise(r => setTimeout(r, 800));
                }
            }
        }
    } catch (err) {
        clearInterval(scrollInterval);
    }
})();
