const fieldOrder = [
  "DRUG_NAME",
  "TARGET",
  "TARGET_PATHWAY",
  "TCGA_DESC",
  "GDSC_TISSUE_DESCRIPTOR_2",
  "CANCER_TYPE_MATCHING_TCGA_LABEL",
  "SITE",
  "GDSC_TISSUE_DESCRIPTOR_1",
];

const contextFields = fieldOrder.filter((field) => field !== "DRUG_NAME");
let lastPrediction = null;
let conversationId = null;

const organRules = [
  {
    organ: "brain",
    label: "Brain",
    terms: ["brain", "nervous", "central_nervous", "glioma", "neuro", "mb", "medulloblastoma", "cns"],
    description: "The view is focused on the cranial and central nervous system area, matching brain or nervous tissue context.",
  },
  {
    organ: "breast",
    label: "Breast tissue",
    terms: ["breast", "brca", "mammary"],
    description: "The chest soft-tissue region is highlighted for breast-associated cancer context.",
  },
  {
    organ: "lungs",
    label: "Lungs",
    terms: ["lung", "nsclc", "sclc", "thoracic", "respiratory", "pleura"],
    description: "The upper chest is emphasized to indicate lung and respiratory tissue involvement.",
  },
  {
    organ: "liver",
    label: "Liver",
    terms: ["liver", "hepatic", "hcc", "lihc", "bile", "biliary"],
    description: "The right upper abdomen is highlighted for liver or hepatobiliary context.",
  },
  {
    organ: "colon",
    label: "Colon",
    terms: ["colon", "colorectal", "large_intestine", "intestine", "bowel", "coad", "read", "digestive"],
    description: "The lower abdomen is focused to represent colon and intestinal tissue involvement.",
  },
  {
    organ: "skin",
    label: "Skin layer",
    terms: ["skin", "melanoma", "skcm", "dermal", "cutaneous"],
    description: "A subtle full-body surface layer is highlighted for skin-associated cancer context.",
  },
];

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

function normalizeText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_");
}

function detectOrgan() {
  const weightedFields = [
    { field: "SITE", weight: 7 },
    { field: "GDSC_TISSUE_DESCRIPTOR_2", weight: 6 },
    { field: "CANCER_TYPE_MATCHING_TCGA_LABEL", weight: 5 },
    { field: "TCGA_DESC", weight: 4 },
    { field: "GDSC_TISSUE_DESCRIPTOR_1", weight: 3 },
    { field: "TARGET_PATHWAY", weight: 2 },
    { field: "TARGET", weight: 1 },
  ];

  let bestMatch = null;

  weightedFields.forEach(({ field, weight }) => {
    const text = normalizeText(document.getElementById(field)?.value || "");
    organRules.forEach((rule) => {
      if (rule.terms.some((term) => text.includes(term))) {
        if (!bestMatch || weight > bestMatch.weight) {
          bestMatch = { ...rule, weight };
        }
      }
    });
  });

  return bestMatch || {
    organ: "default",
    label: "Full body overview",
    description: "Select a cancer type, tissue descriptor, or site to focus the anatomy view and highlight the most relevant body region.",
  };
}

function updateAnatomyFocus() {
  const anatomyStage = document.getElementById("anatomyStage");
  if (!anatomyStage) return;

  const match = detectOrgan();
  const site = document.getElementById("SITE")?.value || "--";
  const tissue =
    document.getElementById("GDSC_TISSUE_DESCRIPTOR_2")?.value ||
    document.getElementById("GDSC_TISSUE_DESCRIPTOR_1")?.value ||
    "--";

  anatomyStage.dataset.organ = match.organ;
  document.getElementById("anatomyState").textContent = match.organ === "default" ? "Full body" : `Focused: ${match.label}`;
  document.getElementById("organName").textContent = match.label;
  document.getElementById("organFocus").textContent =
    match.organ === "default" ? "Awaiting selected cancer context" : `Active focus from ${site}`;
  document.getElementById("organDescription").textContent = match.description;
  document.getElementById("organSite").textContent = `Site: ${site}`;
  document.getElementById("organTissue").textContent = `Tissue: ${tissue}`;
}

function setBusy(isBusy) {
  document.querySelectorAll("button").forEach((button) => {
    button.disabled = isBusy;
  });
}

function formPayload(includeDrug = true) {
  const payload = {};
  const fields = includeDrug ? fieldOrder : contextFields;
  fields.forEach((field) => {
    payload[field] = document.getElementById(field).value;
  });
  return payload;
}

function fillSelect(field, values, currentValue = "") {
  const select = document.getElementById(field);
  select.innerHTML = "";

  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value === "" ? "(blank)" : value;
    select.appendChild(option);
  });

  if (values.includes(currentValue)) {
    select.value = currentValue;
  }
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Request failed: ${response.status}`);
  }
  return data;
}

async function loadStatus() {
  const status = await fetchJson("/api/status");
  document.getElementById("modelStatus").textContent =
    `Regression: ${status.regression_model} | Classification: ${status.classification_model} | Drugs: ${status.available_drugs}`;
}

async function loadOptions() {
  const current = Object.fromEntries(fieldOrder.map((field) => [field, document.getElementById(field)?.value || ""]));
  const [data, sample] = await Promise.all([fetchJson("/api/options"), fetchJson("/api/sample-input")]);
  data.field_order.forEach((field) => fillSelect(field, data.fields[field] || [], current[field]));
  fieldOrder.forEach((field) => {
    const select = document.getElementById(field);
    if (!current[field] && sample[field] !== undefined) {
      select.value = sample[field];
    }
  });
  updateAnatomyFocus();
}

function updateMetrics(result) {
  document.getElementById("ic50Value").textContent = formatNumber(result.Predicted_IC50, 5);
  document.getElementById("lnValue").textContent = formatNumber(result.Predicted_LN_IC50, 5);
  document.getElementById("sensitivityValue").textContent = result.Sensitivity || "--";
  document.getElementById("aucValue").textContent = formatNumber(result.Used_AUC, 5);
}

function renderRecommendations(rows) {
  const body = document.getElementById("recommendationsBody");
  body.innerHTML = "";
  document.getElementById("resultCount").textContent = `${rows.length} rows`;

  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="6" class="empty">No matching recommendations.</td></tr>';
    return;
  }

  rows.forEach((row, index) => {
    const tr = document.createElement("tr");
    const cls = row.Sensitivity === "Resistant" ? "bad" : "good";
    tr.innerHTML = `
      <td>${index + 1}</td>
      <td>${row.DRUG_NAME}</td>
      <td><span class="pill ${cls}">${row.Sensitivity}</span></td>
      <td>${formatNumber(row.Predicted_IC50, 5)}</td>
      <td>${formatNumber(row.Predicted_LN_IC50, 5)}</td>
      <td>${formatNumber(row.Used_AUC, 5)}</td>
    `;
    body.appendChild(tr);
  });
}

async function predict() {
  setBusy(true);
  try {
    const result = await fetchJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formPayload(true)),
    });
    lastPrediction = result;
    updateMetrics(result);
    renderRecommendations([result]);
  } catch (error) {
    alert(error.message);
  } finally {
    setBusy(false);
  }
}

async function recommend() {
  setBusy(true);
  try {
    const payload = formPayload(false);
    const topN = document.getElementById("top_n").value;
    if (topN) payload.top_n = Number(topN);

    const result = await fetchJson("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderRecommendations(result.recommendations);
    if (result.recommendations[0]) {
      lastPrediction = result.recommendations[0];
      updateMetrics(result.recommendations[0]);
    }
  } catch (error) {
    alert(error.message);
  } finally {
    setBusy(false);
  }
}

function appendMessage(role, content) {
  const log = document.getElementById("chatLog");
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = content;
  log.appendChild(message);
  log.scrollTop = log.scrollHeight;
}

async function sendChat() {
  const input = document.getElementById("chatInput");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  appendMessage("user", text);

  try {
    const result = await fetchJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        conversation_id: conversationId,
        context: lastPrediction,
      }),
    });
    conversationId = result.conversation_id;
    document.getElementById("chatState").textContent = result.model_used || "ready";
    appendMessage("assistant", result.message);
  } catch (error) {
    appendMessage("assistant", error.message);
  }
}

document.getElementById("predictBtn").addEventListener("click", predict);
document.getElementById("recommendBtn").addEventListener("click", recommend);
document.getElementById("reloadOptions").addEventListener("click", loadOptions);
document.getElementById("chatBtn").addEventListener("click", sendChat);
contextFields.forEach((field) => {
  document.getElementById(field).addEventListener("change", updateAnatomyFocus);
});
document.getElementById("chatInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") sendChat();
});

async function boot() {
  setBusy(true);
  try {
    await Promise.all([loadStatus(), loadOptions()]);
  } catch (error) {
    document.getElementById("modelStatus").textContent = error.message;
  } finally {
    setBusy(false);
  }
}

boot();
