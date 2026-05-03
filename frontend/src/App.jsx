import { useEffect, useMemo, useState } from "react";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function App() {
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [annotatedUrl, setAnnotatedUrl] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [streamOn, setStreamOn] = useState(false);
  const [conf, setConf] = useState(0.45);
  const [iou, setIou] = useState(0.5);

  const streamUrl = useMemo(() => {
    const params = new URLSearchParams({
      conf: String(conf),
      iou: String(iou),
      alarm_threshold: "5"
    });
    return `${apiBase}/camera/stream?${params.toString()}`;
  }, [apiBase, conf, iou]);

  const onSelectFile = (event) => {
    const file = event.target.files?.[0];
    setSelectedFile(file ?? null);
    setPrediction(null);
    setAnnotatedUrl("");
    setError("");
    if (file) {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
      const localUrl = URL.createObjectURL(file);
      setPreviewUrl(localUrl);
    } else {
      setPreviewUrl("");
    }
  };

  const runImageDetection = async () => {
    if (!selectedFile) {
      setError("Please choose an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setPrediction(null);
    setAnnotatedUrl("");

    try {
      const createFormData = () => {
        const formData = new FormData();
        formData.append("file", selectedFile);
        return formData;
      };

      const query = `conf=${encodeURIComponent(conf)}&iou=${encodeURIComponent(iou)}`;
      const [jsonResponse, imageResponse] = await Promise.all([
        fetch(`${apiBase}/predict/image?${query}`, {
          method: "POST",
          body: createFormData()
        }),
        fetch(`${apiBase}/predict/image/annotated?${query}`, {
          method: "POST",
          body: createFormData()
        })
      ]);

      if (!jsonResponse.ok) {
        const text = await jsonResponse.text();
        throw new Error(`Predict JSON failed: ${jsonResponse.status} ${text}`);
      }

      if (!imageResponse.ok) {
        const text = await imageResponse.text();
        throw new Error(`Predict image failed: ${imageResponse.status} ${text}`);
      }

      const jsonData = await jsonResponse.json();
      const imageBlob = await imageResponse.blob();
      const imageUrl = URL.createObjectURL(imageBlob);

      if (annotatedUrl) {
        URL.revokeObjectURL(annotatedUrl);
      }
      setPrediction(jsonData);
      setAnnotatedUrl(imageUrl);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown request error.");
    } finally {
      setLoading(false);
    }
  };

  const streamStatusText = streamOn ? "Streaming from backend camera..." : "Stream paused";

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
      if (annotatedUrl) {
        URL.revokeObjectURL(annotatedUrl);
      }
    };
  }, [previewUrl, annotatedUrl]);

  return (
    <div className="page">
      <div className="orb orb-a" />
      <div className="orb orb-b" />
      <main className="container">
        <header className="header">
          <h1>Driver Detection Dashboard</h1>
          <p>Upload image hoặc xem realtime webcam stream từ FastAPI backend.</p>
        </header>

        <section className="panel control-panel">
          <div className="field">
            <label htmlFor="apiBase">Backend API</label>
            <input
              id="apiBase"
              value={apiBase}
              onChange={(e) => setApiBase(e.target.value.trim())}
              placeholder="http://127.0.0.1:8000"
            />
          </div>
          <div className="range-row">
            <div className="field">
              <label htmlFor="conf">Conf ({conf.toFixed(2)})</label>
              <input
                id="conf"
                type="range"
                min="0.1"
                max="0.95"
                step="0.01"
                value={conf}
                onChange={(e) => setConf(Number(e.target.value))}
              />
            </div>
            <div className="field">
              <label htmlFor="iou">IoU ({iou.toFixed(2)})</label>
              <input
                id="iou"
                type="range"
                min="0.1"
                max="0.95"
                step="0.01"
                value={iou}
                onChange={(e) => setIou(Number(e.target.value))}
              />
            </div>
          </div>
        </section>

        <section className="grid">
          <article className="panel">
            <div className="panel-head">
              <h2>1) Upload Image Test</h2>
            </div>
            <div className="field">
              <input type="file" accept="image/*" onChange={onSelectFile} />
            </div>
            <button className="btn primary" onClick={runImageDetection} disabled={loading}>
              {loading ? "Detecting..." : "Run Detection"}
            </button>
            {error ? <p className="error">{error}</p> : null}

            <div className="two-cols">
              <div>
                <h3>Original</h3>
                <div className="frame">
                  {previewUrl ? <img src={previewUrl} alt="original upload" /> : <p>No image</p>}
                </div>
              </div>
              <div>
                <h3>Annotated (Top-1)</h3>
                <div className="frame">
                  {annotatedUrl ? <img src={annotatedUrl} alt="annotated output" /> : <p>No result</p>}
                </div>
              </div>
            </div>
          </article>

          <article className="panel">
            <div className="panel-head">
              <h2>2) Realtime Camera Test</h2>
              <button className="btn" onClick={() => setStreamOn((v) => !v)}>
                {streamOn ? "Stop Stream" : "Start Stream"}
              </button>
            </div>
            <p className="status">{streamStatusText}</p>
            <div className="frame stream-frame">
              {streamOn ? (
                <img src={streamUrl} alt="camera stream" />
              ) : (
                <p>Click Start Stream để xem /camera/stream.</p>
              )}
            </div>
          </article>
        </section>
      </main>
    </div>
  );
}

export default App;
