const textEl = document.getElementById("text");
const apiUrlEl = document.getElementById("apiUrl");
const btn = document.getElementById("btn");

const resultEl = document.getElementById("result");
const statusEl = document.getElementById("status");
const scoreEl = document.getElementById("score");
const threshEl = document.getElementById("thresh");

const STORAGE_KEY = "toxic_or_not_api_url";

function normalizeBaseUrl(url) {
    return url.replace(/\/+$/, "");
}

function setLoading(isLoading) {
    btn.disabled = isLoading;
    btn.textContent = isLoading ? "Predicting..." : "Predict";
}

function showError(message) {
    resultEl.classList.remove("hidden");
    statusEl.textContent = `Error: ${message}`;
    scoreEl.textContent = "-";
    threshEl.textContent = "-";
}

function showResult({ toxicity_score, is_toxic, threshold }) {
    resultEl.classList.remove("hidden");
    statusEl.textContent = is_toxic ? "TOXIC" : "NOT TOXIC";
    scoreEl.textContent = toxicity_score.toFixed(4);
    threshEl.textContent = Number(threshold).toFixed(2);
}

apiUrlEl.value = localStorage.getItem(STORAGE_KEY) || "https://toxic-or-not-1.onrender.com";

btn.addEventListener("click", async () => {
    const text = textEl.value.trim();
    if (!text) return showError("Please enter some text.");

    const baseUrlRaw = apiUrlEl.value.trim();
    if (!baseUrlRaw) return showError("Please enter your API URL (Render service).");

    const baseUrl = normalizeBaseUrl(baseUrlRaw);
    localStorage.setItem(STORAGE_KEY, baseUrl);

    setLoading(true);
    try {
        const res = await fetch(`${baseUrl}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        if (!res.ok) {
            const body = await res.text();
            throw new Error(`${res.status} ${res.statusText} — ${body}`);
        }

        const data = await res.json();
        showResult(data);
    } catch (err) {
        showError(err.message || String(err));
    } finally {
        setLoading(false);
    }
});
