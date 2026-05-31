// ========= Splash Screen =========
(function() {
  const splash = document.getElementById("splashScreen");
  const app = document.getElementById("appContainer");
  if (splash && app) {
    setTimeout(() => {
      splash.classList.add("fade-out");
      app.style.display = "flex";
      setTimeout(() => {
        splash.remove();
      }, 500);
    }, 2000);
  }
})();

// Global state variables
let analyticsChart = null;
let selectedFile = null;

// Sidebar Navigation
const links = document.querySelectorAll(".nav-link");
const screens = document.querySelectorAll(".screen");

links.forEach(link => {
  link.addEventListener("click", () => {
    // Remove active class
    links.forEach(item => {
      item.classList.remove("active");
    });

    // Add active class
    link.classList.add("active");

    // Hide screens
    screens.forEach(screen => {
      screen.classList.remove("active-screen");
    });

    // Show selected screen
    const screenId = link.getAttribute("data-screen");
    document.getElementById(screenId).classList.add("active-screen");

    // Fetch screen-specific data
    loadScreenData(screenId);
  });
});

// Navigation Data Loader Router
function loadScreenData(screenId) {
  switch (screenId) {
    case "dashboard":
      loadDashboardStats();
      break;
    case "analytics":
      loadAnalyticsData();
      break;
    case "gallery":
      loadGallery();
      break;
    case "history":
      loadHistory();
      break;
    case "notifications":
      loadNotifications();
      break;
    case "settings":
      loadSettings();
      break;
    default:
      break;
  }
}

// 1. Dashboard Stats
function loadDashboardStats() {
  fetch('/api/stats')
    .then(res => res.json())
    .then(data => {
      const cards = document.querySelectorAll("#dashboard .card");
      if (cards.length >= 4) {
        cards[0].querySelector("p").innerText = data.totalUploads;
        cards[1].querySelector("p").innerText = data.aiProcessed;
        cards[2].querySelector("p").innerText = data.reports;
        cards[3].querySelector("p").innerText = data.accuracy;
      }
    })
    .catch(err => console.error("Error loading stats:", err));
}

// 2. Chart / Analytics
function initChart(labels, dataPoints) {
  const ctx = document.getElementById("analyticsChart");
  if (!ctx) return;
  
  if (analyticsChart) {
    analyticsChart.destroy();
  }

  analyticsChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [{
        label: "AI Analytics",
        data: dataPoints,
        borderWidth: 3,
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59, 130, 246, 0.1)",
        fill: true,
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          labels: {
            color: "white",
            font: { family: 'Poppins' }
          }
        }
      },
      scales: {
        y: {
          grid: { color: "rgba(255, 255, 255, 0.1)" },
          ticks: {
            color: "white",
            font: { family: 'Poppins' }
          }
        },
        x: {
          grid: { color: "rgba(255, 255, 255, 0.1)" },
          ticks: {
            color: "white",
            font: { family: 'Poppins' }
          }
        }
      }
    }
  });
}

function loadAnalyticsData() {
  fetch('/api/analytics')
    .then(res => res.json())
    .then(data => {
      initChart(data.labels, data.data);
    })
    .catch(err => console.error("Error loading analytics:", err));
}

// 3. Image Upload and Analysis
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const previewImage = document.getElementById("previewImage");
const analyzeBtn = document.getElementById("analyzeBtn");
const loadingSpinner = document.getElementById("loadingSpinner");
const analysisResult = document.getElementById("analysisResult");

imageInput.addEventListener("change", function(e) {
  const file = e.target.files[0];
  if (file) {
    selectedFile = file;
    preview.style.display = "block";
    previewImage.src = URL.createObjectURL(file);
    analyzeBtn.style.display = "block";
    document.getElementById("plantDescSection").style.display = "block";
    
    // Hide old results
    analysisResult.style.display = "none";
  }
});

analyzeBtn.addEventListener("click", () => {
  if (!selectedFile) return;

  // Toggle UI states
  analyzeBtn.style.display = "none";
  loadingSpinner.style.display = "block";
  analysisResult.style.display = "none";

  const formData = new FormData();
  formData.append("image", selectedFile);
  const plantDesc = document.getElementById("plantDescription").value.trim();
  if (plantDesc) formData.append("description", plantDesc);
  fetch('/api/upload', {
    method: 'POST',
    body: formData
  })
    .then(res => {
      if (!res.ok) {
        return res.json().then(err => { throw new Error(err.error || "Upload failed") });
      }
      return res.json();
    })
    .then(data => {
      loadingSpinner.style.display = "none";
      analysisResult.style.display = "block";

      // Render details
      document.getElementById("resultCropDisease").innerText = `${data.crop} - ${data.disease}`;
      document.getElementById("resultConfidence").innerText = `${(data.confidence * 100).toFixed(0)}%`;
      document.getElementById("resultDescription").innerText = data.description;

      const statusBadge = document.getElementById("resultStatusBadge");
      statusBadge.innerText = data.status;
      statusBadge.className = `status-badge ${data.status.toLowerCase()}`;

      const recsList = document.getElementById("resultRecommendations");
      recsList.innerHTML = data.recommendations.map(rec => `<li>${rec}</li>`).join('');

      const warningEl = document.getElementById("resultWarning");
      if (data.status && data.status.toLowerCase() === "diseased") {
        const sev = data.severity || "Moderate";
        const sevColor = sev === "Critical" ? "#dc2626" : sev === "High" ? "#ef4444" : sev === "Moderate" ? "#f59e0b" : "#10b981";
        warningEl.innerHTML = `<span style="font-size:18px;">⚠️</span> <strong>${sev} Severity</strong> — ${data.disease} detected on ${data.crop}. Immediate treatment recommended. Follow the solutions below.`;
        warningEl.style.display = "block";
        warningEl.style.background = sevColor + "22";
        warningEl.style.borderLeft = "4px solid " + sevColor;
        warningEl.style.color = "#fbbf24";
        warningEl.style.padding = "12px 16px";
        warningEl.style.borderRadius = "8px";
        warningEl.style.marginBottom = "12px";
        warningEl.style.fontSize = "14px";
      } else {
        warningEl.style.display = "none";
      }

      // Draw bounding box if multi-model pipeline returned one
      const canvas = document.getElementById("bboxCanvas");
      const bboxLabel = document.getElementById("bboxLabel");
      if (canvas && data.bounding_box) {
        const img = document.getElementById("previewImage");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const [x1p, y1p, x2p, y2p] = data.bounding_box;
        const x1 = (x1p / 100) * canvas.width;
        const y1 = (y1p / 100) * canvas.height;
        const x2 = (x2p / 100) * canvas.width;
        const y2 = (y2p / 100) * canvas.height;
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 4]);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        if (bboxLabel) {
          bboxLabel.style.display = "block";
          bboxLabel.innerText = `🔬 YOLOv11: Leaf Detected`;
        }
      } else if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (bboxLabel) bboxLabel.style.display = "none";
      }
    })
    .catch(err => {
      loadingSpinner.style.display = "none";
      analyzeBtn.style.display = "block";
      alert("Error analyzing image: " + err.message);
    });
});

// 4. Gallery Screen
let galleryData = [];

function loadGallery() {
  const galleryGrid = document.querySelector(".gallery-grid");
  if (!galleryGrid) return Promise.resolve();

  return fetch('/api/gallery')
    .then(res => res.json())
    .then(data => {
      galleryData = data;
      
      if (!data || data.length === 0) {
        galleryGrid.innerHTML = `
          <div style="grid-column: 1 / -1; text-align: center; padding: 50px 20px; background: rgba(30, 41, 59, 0.4); border-radius: 20px; border: 2px dashed #334155;">
            <div style="font-size: 50px; margin-bottom: 20px;">🖼️</div>
            <h3 style="color: white; margin-bottom: 10px;">No images uploaded yet</h3>
            <p style="color: #94a3b8;">Please upload plant or animal images in the respective sections to see them here in the gallery.</p>
          </div>
        `;
        return;
      }

      galleryGrid.innerHTML = data.map((item, index) => `
        <div class="card gallery-card" data-index="${index}" style="padding: 10px; display: flex; flex-direction: column; gap: 10px; text-align: left; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s;">
          <img src="${item.url}" style="width: 100%; border-radius: 14px; height: 180px; object-fit: cover;">
          <div style="padding: 5px;">
            <h3 style="font-size: 16px; color: white;">${item.type === 'animal' ? (item.animal || item.crop) : item.crop}</h3>
            <p style="font-size: 13px; color: #94a3b8; margin-top: 4px;">${item.type === 'animal' ? (item.diagnosis || item.disease) : item.disease}</p>
            <span class="status-badge ${item.status.toLowerCase()}" style="display: inline-block; margin-top: 8px; font-size: 11px; padding: 3px 8px;">${item.type === 'animal' ? (item.severity || item.status) : item.status}</span>
          </div>
        </div>
      `).join('');

      // Attach click handlers
      document.querySelectorAll(".gallery-card").forEach(card => {
        card.addEventListener("click", () => {
          const idx = parseInt(card.getAttribute("data-index"));
          showGalleryDetail(galleryData[idx]);
        });
      });
    })
    .catch(err => console.error("Error loading gallery:", err));
}

function showDeleteToast(message) {
  const toast = document.createElement("div");
  toast.innerText = message;
  toast.style.cssText = "position:fixed;bottom:30px;left:50%;transform:translateX(-50%);background:#10b981;color:white;padding:12px 24px;border-radius:10px;font-size:14px;font-weight:600;z-index:99999;box-shadow:0 4px 20px rgba(0,0,0,0.3);transition:opacity 0.4s;";
  document.body.appendChild(toast);
  setTimeout(() => { toast.style.opacity = "0"; }, 2000);
  setTimeout(() => { toast.remove(); }, 2500);
}

function showGalleryDetail(item) {
  const modal = document.getElementById("galleryDetailModal");
  if (!modal) return;

  document.getElementById("modalImage").src = item.url;
  document.getElementById("modalTimestamp").innerText = item.timestamp ? `Analyzed: ${item.timestamp}` : "";
  document.getElementById("modalFilename").innerText = item.filename ? `File: ${item.filename}` : "";

  const plantContent = document.getElementById("modalPlantContent");
  const animalContent = document.getElementById("modalAnimalContent");

  if (item.type === "animal") {
    plantContent.style.display = "none";
    animalContent.style.display = "block";

    document.getElementById("modalAnimalTitle").innerText = `${item.animal || item.crop} - ${item.diagnosis || item.disease}`;
    const severityBadge = document.getElementById("modalSeverityBadge");
    const sev = item.severity || item.status || "Moderate";
    severityBadge.innerText = sev;
    severityBadge.className = `status-badge ${sev.toLowerCase() === 'high' ? 'diseased' : sev.toLowerCase() === 'low' ? 'healthy' : 'diseased'}`;
    document.getElementById("modalAnimalDescription").innerText = item.description || "No description available.";
    const solList = document.getElementById("modalAnimalSolutions");
    const solutions = item.solution || item.recommendations || [];
    solList.innerHTML = solutions.map(s => `<li>${s}</li>`).join('');
  } else {
    animalContent.style.display = "none";
    plantContent.style.display = "block";

    document.getElementById("modalCropDisease").innerText = `${item.crop} - ${item.disease}`;
    const statusBadge = document.getElementById("modalStatusBadge");
    statusBadge.innerText = item.status;
    statusBadge.className = `status-badge ${item.status.toLowerCase()}`;
    document.getElementById("modalConfidence").innerText = item.confidence ? `Confidence: ${(item.confidence * 100).toFixed(0)}%` : "";
    document.getElementById("modalDescription").innerText = item.description || "No description available.";
    const recsList = document.getElementById("modalRecommendations");
    const recs = item.recommendations || [];
    recsList.innerHTML = recs.map(r => `<li>${r}</li>`).join('');
  }

  // Set up Delete Diagnosis button handler
  const deleteBtn = document.getElementById("deleteGalleryItemBtn");
  if (deleteBtn) {
    const newDeleteBtn = deleteBtn.cloneNode(true);
    newDeleteBtn.disabled = false;
    newDeleteBtn.innerHTML = "🗑️ Delete Diagnosis";
    deleteBtn.parentNode.replaceChild(newDeleteBtn, deleteBtn);
    
    newDeleteBtn.addEventListener("click", () => {
      if (confirm("Are you sure you want to delete this diagnosis record?")) {
        newDeleteBtn.disabled = true;
        newDeleteBtn.innerHTML = "⏳ Deleting...";
        
        const deleteId = item.id || item.filename;
        fetch(`/api/history/${deleteId}`, {
          method: 'DELETE'
        })
        .then(res => {
          if (res.ok) return res.json();
          throw new Error("Could not delete record from server");
        })
        .then(resData => {
          // Close modal immediately (no blocking alert)
          modal.style.display = "none";
          document.body.style.overflow = "auto";

          // Show non-blocking toast notification
          showDeleteToast("✅ Diagnosis deleted successfully!");

          // Wait for gallery to fully refresh before user can interact
          return loadGallery();
        })
        .catch(err => {
          alert("Error deleting diagnosis: " + err.message);
          newDeleteBtn.disabled = false;
          newDeleteBtn.innerHTML = "🗑️ Delete Diagnosis";
        });
      }
    });
  }

  modal.style.display = "block";
  document.body.style.overflow = "hidden";
}

// Close gallery modal
document.addEventListener("DOMContentLoaded", () => {
  const closeBtn = document.getElementById("closeGalleryModal");
  const modal = document.getElementById("galleryDetailModal");
  if (closeBtn && modal) {
    closeBtn.addEventListener("click", () => {
      modal.style.display = "none";
      document.body.style.overflow = "";
    });
    modal.addEventListener("click", (e) => {
      if (e.target === modal) {
        modal.style.display = "none";
        document.body.style.overflow = "";
      }
    });
  }
});

// 5. History Screen
function loadHistory() {
  const historyActivity = document.querySelector("#history .activity");
  if (!historyActivity) return;

  fetch('/api/history')
    .then(res => res.json())
    .then(data => {
      historyActivity.innerHTML = data.map(item => `
        <div class="activity-item" style="display: flex; justify-content: space-between; align-items: center; text-align: left;">
          <div>
            <h3 style="font-size: 16px; color: white;">${item.crop} - ${item.disease}</h3>
            <p style="color: #64748b; font-size: 12px; margin-top: 4px;">File: ${item.filename} | Analyzed on ${item.timestamp}</p>
          </div>
          <div style="text-align: right;">
            <div style="font-size: 14px; font-weight: bold; color: #3b82f6;">Confidence: ${(item.confidence * 100).toFixed(0)}%</div>
            <span class="status-badge ${item.status.toLowerCase()}" style="display: inline-block; margin-top: 6px; font-size: 11px; padding: 2px 6px;">${item.status}</span>
          </div>
        </div>
      `).join('');
    })
    .catch(err => console.error("Error loading history:", err));
}

// 6. Settings Screen (Cleaned up - Hugging Face now runs automatically backend-side)
function loadSettings() {
  // Only appearance settings are handled on the frontend now
}


// 7. Download Report Button (Generates beautiful PDF of full history instead of JSON)
const downloadReportBtn = document.getElementById("downloadReportBtn");
if (downloadReportBtn) {
  downloadReportBtn.addEventListener("click", () => {
    generateReport(365, "Full Farm Diagnostic History");
  });
}

// 8. Notifications Screen
function loadNotifications() {
  const container = document.getElementById("notificationsList");
  if (!container) return;

  fetch('/api/history')
    .then(res => res.json())
    .then(data => {
      if (!data || data.length === 0) {
        container.innerHTML = `
          <div class="activity-item">
            <h3>No notifications yet</h3>
            <p>Upload leaf images in the Uploads section to get started.</p>
          </div>
        `;
        return;
      }

      // Generate dynamic notifications based on history
      container.innerHTML = data.map(item => {
        const isDiseased = item.status.toLowerCase() === 'diseased';
        const title = isDiseased 
          ? `⚠️ Critical Alert: ${item.crop} Disease Detected`
          : `✅ Analysis Completed: ${item.crop} Leaf`;
        const message = isDiseased
          ? `Plant leaf identified with ${item.disease} (${(item.confidence * 100).toFixed(0)}% confidence) at ${item.timestamp}. Immediate treatment recommended.`
          : `Plant leaf analyzed successfully at ${item.timestamp} and verified Healthy with ${(item.confidence * 100).toFixed(0)}% confidence.`;

        return `
          <div class="activity-item" style="border-left: 4px solid ${isDiseased ? '#ef4444' : '#10b981'}; padding-left: 15px; margin-bottom: 15px; text-align: left;">
            <h3 style="font-size: 16px; color: white; margin: 0 0 5px 0;">${title}</h3>
            <p style="color: #94a3b8; font-size: 13px; margin: 0 0 4px 0;">${message}</p>
            <span style="color: #64748b; font-size: 11px;">File: ${item.filename}</span>
          </div>
        `;
      }).join('');
    })
    .catch(err => console.error("Error loading notifications:", err));
}

// 9. Report Downloads Handler (Generates Premium styled Print-to-PDF Report)
function generateReport(daysRange, reportName) {
  fetch('/api/history')
    .then(res => res.json())
    .then(data => {
      const now = new Date();
      const cutoffDate = new Date();
      cutoffDate.setDate(now.getDate() - daysRange);

      // Filter history entries by timestamp range
      const filtered = data.filter(item => {
        try {
          const itemDate = new Date(item.timestamp.replace(' ', 'T'));
          return itemDate >= cutoffDate;
        } catch (e) {
          return true; // fallback
        }
      });

      // Calculate stats
      const total = filtered.length;
      const diseasedCount = filtered.filter(item => item.status && item.status.toLowerCase() === 'diseased').length;
      const healthyCount = total - diseasedCount;
      const diseasedPct = total > 0 ? ((diseasedCount / total) * 100).toFixed(0) : 0;
      const healthyPct = total > 0 ? ((healthyCount / total) * 100).toFixed(0) : 0;

      // Open new window to print beautifully
      const printWindow = window.open("", "_blank");
      printWindow.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
          <title>${reportName}</title>
          <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
            body { 
              font-family: 'Outfit', sans-serif; 
              padding: 40px; 
              color: #0f172a; 
              background: #ffffff;
              line-height: 1.5;
            }
            .header-container {
              display: flex;
              justify-content: space-between;
              align-items: center;
              border-bottom: 2px solid #10b981;
              padding-bottom: 20px;
              margin-bottom: 30px;
            }
            .logo-section h1 {
              color: #059669;
              font-size: 28px;
              margin: 0;
              font-weight: 700;
              letter-spacing: -0.5px;
            }
            .logo-section p {
              color: #64748b;
              margin: 4px 0 0 0;
              font-size: 14px;
            }
            .meta-section {
              text-align: right;
              font-size: 13px;
              color: #475569;
            }
            .stats-grid {
              display: grid;
              grid-template-columns: repeat(3, 1fr);
              gap: 20px;
              margin-bottom: 40px;
            }
            .stat-card {
              background: #f8fafc;
              border: 1px solid #e2e8f0;
              border-radius: 12px;
              padding: 20px;
              text-align: center;
              box-shadow: 0 1px 3px rgba(0,0,0,0.02);
            }
            .stat-card h3 {
              color: #64748b;
              font-size: 12px;
              font-weight: 600;
              text-transform: uppercase;
              letter-spacing: 0.5px;
              margin: 0 0 8px 0;
            }
            .stat-card .value {
              font-size: 32px;
              font-weight: 700;
              color: #0f172a;
            }
            .stat-card .desc {
              font-size: 13px;
              color: #64748b;
              margin-top: 4px;
            }
            h2 {
              font-size: 18px;
              color: #1e293b;
              border-bottom: 1px solid #e2e8f0;
              padding-bottom: 8px;
              margin-bottom: 16px;
              font-weight: 600;
            }
            table {
              width: 100%;
              border-collapse: collapse;
              margin-top: 10px;
            }
            th {
              background: #f1f5f9;
              color: #475569;
              font-weight: 600;
              text-align: left;
              padding: 12px 16px;
              font-size: 13px;
              border-bottom: 2px solid #e2e8f0;
            }
            td {
              padding: 12px 16px;
              border-bottom: 1px solid #e2e8f0;
              font-size: 13px;
              color: #334155;
            }
            tr:nth-child(even) td {
              background: #f8fafc;
            }
            .badge {
              display: inline-block;
              padding: 4px 8px;
              border-radius: 6px;
              font-size: 11px;
              font-weight: 600;
              text-transform: uppercase;
            }
            .badge.healthy {
              background: #dcfce7;
              color: #15803d;
            }
            .badge.diseased {
              background: #fee2e2;
              color: #b91c1c;
            }
            .footer {
              margin-top: 60px;
              border-top: 1px solid #e2e8f0;
              padding-top: 15px;
              text-align: center;
              font-size: 11px;
              color: #94a3b8;
            }
            @media print {
              body { padding: 0; }
              button { display: none; }
            }
          </style>
        </head>
        <body>
          <div class="header-container">
            <div class="logo-section">
              <h1>🌿 AgroAI Crop Health Report</h1>
              <p>Intelligent Farm Diagnostic Monitoring System</p>
            </div>
            <div class="meta-section">
              <strong>Report:</strong> ${reportName}<br>
              <strong>Date:</strong> ${now.toLocaleDateString()}<br>
              <strong>Time:</strong> ${now.toLocaleTimeString()}
            </div>
          </div>
          
          <div class="stats-grid">
            <div class="stat-card" style="border-top: 4px solid #3b82f6;">
              <h3>Total Analyzed</h3>
              <div class="value">${total}</div>
              <div class="desc">Leaf uploads / Scans</div>
            </div>
            <div class="stat-card" style="border-top: 4px solid #10b981;">
              <h3>Healthy</h3>
              <div class="value">${healthyCount}</div>
              <div class="desc">${healthyPct}% of total scans</div>
            </div>
            <div class="stat-card" style="border-top: 4px solid #ef4444;">
              <h3>Diseased</h3>
              <div class="value">${diseasedCount}</div>
              <div class="desc">${diseasedPct}% of total scans</div>
            </div>
          </div>
          
          <h2>📋 Detailed Diagnostics Log</h2>
          <table>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Diagnostic Type</th>
                <th>Subject Name</th>
                <th>Condition Diagnosis</th>
                <th>Severity Status</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              ${total === 0 ? `
                <tr>
                  <td colspan="6" style="text-align: center; color: #64748b;">No diagnostic entries found for the selected period.</td>
                </tr>
              ` : filtered.map(item => `
                <tr>
                  <td>${item.timestamp}</td>
                  <td>${(item.type || "plant").toUpperCase()}</td>
                  <td><strong>${item.crop || item.animal || "Plant"}</strong></td>
                  <td>${item.disease || item.diagnosis || "Healthy"}</td>
                  <td><span class="badge ${item.status && item.status.toLowerCase() === 'healthy' ? 'healthy' : 'diseased'}">${item.status || "Diseased"}</span></td>
                  <td>${item.confidence ? (item.confidence * 100).toFixed(0) + '%' : '85%'}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
          
          <div class="footer">
            AgroAI Crop Pathology System &copy; 2026. All rights reserved. Generated automatically via secure backend integration.
          </div>
          
          <script>
            window.onload = function() {
              setTimeout(function() {
                window.print();
                window.close();
              }, 600);
            }
          </script>
        </body>
        </html>
      `);
      printWindow.document.close();
    })
    .catch(err => alert("Error generating report: " + err.message));
}

const downloadWeeklyBtn = document.getElementById("downloadWeeklyBtn");
if (downloadWeeklyBtn) {
  downloadWeeklyBtn.addEventListener("click", () => {
    generateReport(7, "Weekly Crop Diagnostics Summary");
  });
}

const downloadMonthlyBtn = document.getElementById("downloadMonthlyBtn");
if (downloadMonthlyBtn) {
  downloadMonthlyBtn.addEventListener("click", () => {
    generateReport(30, "Monthly Crop Diagnostics Summary");
  });
}

// --- 4. Animal Health Diagnostics & Nearby Vets Finder ---
const animalImageInput = document.getElementById("animalImageInput");
const animalPreview = document.getElementById("animalPreview");
const animalPreviewImage = document.getElementById("animalPreviewImage");
const analyzeAnimalBtn = document.getElementById("analyzeAnimalBtn");
const animalLoadingSpinner = document.getElementById("animalLoadingSpinner");
const animalResult = document.getElementById("animalResult");

let selectedAnimalFile = null;

if (animalImageInput) {
  animalImageInput.addEventListener("change", function(e) {
    const file = e.target.files[0];
    if (file) {
      selectedAnimalFile = file;
      animalPreview.style.display = "block";
      animalPreviewImage.src = URL.createObjectURL(file);
      analyzeAnimalBtn.style.display = "block";
      document.getElementById("animalDescSection").style.display = "block";
      animalResult.style.display = "none";
    }
  });
}

if (analyzeAnimalBtn) {
  analyzeAnimalBtn.addEventListener("click", () => {
    if (!selectedAnimalFile) return;

    analyzeAnimalBtn.style.display = "none";
    animalLoadingSpinner.style.display = "block";
    animalResult.style.display = "none";

    const formData = new FormData();
    formData.append("image", selectedAnimalFile);
    const animalDesc = document.getElementById("animalDescription").value.trim();
    if (animalDesc) formData.append("description", animalDesc);

    fetch('/api/animal/upload', {
      method: 'POST',
      body: formData
    })
      .then(res => {
        if (!res.ok) {
          return res.json().then(err => { throw new Error(err.error || "Animal analysis failed") });
        }
        return res.json();
      })
      .then(data => {
        animalLoadingSpinner.style.display = "none";
        animalResult.style.display = "block";

        // Render diagnosis
        document.getElementById("animalResultTitle").innerText = `${data.animal} - ${data.diagnosis}`;
        document.getElementById("animalResultDescription").innerText = data.description;

        const severityBadge = document.getElementById("animalSeverityBadge");
        severityBadge.innerText = data.severity;
        severityBadge.className = `status-badge ${data.severity.toLowerCase() === 'low' ? 'healthy' : 'diseased'}`;

        const solutionsList = document.getElementById("animalResultRecommendations");
        solutionsList.innerHTML = data.solution.map(sol => `<li>${sol}</li>`).join('');

        // Map and Vet Care - always show after diagnosis
        const vetContainer = document.getElementById("vetHospitalContainer");
        if (data.major) {
          vetContainer.querySelector("h3").innerHTML = "⚠️ Major Issue Detected - Nearby Veterinary Care Found";
        } else {
          vetContainer.querySelector("h3").innerHTML = "🏥 Nearby Veterinary Clinics";
        }
        vetContainer.style.display = "block";
        initializeVetFinder();
      })
      .catch(err => {
        animalLoadingSpinner.style.display = "none";
        analyzeAnimalBtn.style.display = "block";
        alert(err.message);
      });
  });
}

function initializeVetFinder() {
  const hospitalList = document.getElementById("vetHospitalList");
  if (hospitalList) {
    hospitalList.innerHTML = `<p style="color: #94a3b8; font-size: 14px;">📍 Getting your location...</p>`;
  }

  // Setup manual search button bindings
  const searchBtn = document.getElementById("searchLocationBtn");
  const manualInput = document.getElementById("manualLocationInput");
  
  if (searchBtn && manualInput && !searchBtn.dataset.bound) {
    searchBtn.dataset.bound = "true";
    
    const performSearch = () => {
      const query = manualInput.value.trim();
      if (!query) return;
      
      if (hospitalList) {
        hospitalList.innerHTML = `<p style="color: #94a3b8; font-size: 14px;">🔍 Searching for location coordinates...</p>`;
      }
      
      fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}`)
        .then(res => res.json())
        .then(data => {
          if (data && data.length > 0) {
            const firstResult = data[0];
            const lat = parseFloat(firstResult.lat);
            const lon = parseFloat(firstResult.lon);
            showVetHospitals(lat, lon);
          } else {
            if (hospitalList) {
              hospitalList.innerHTML = `<p style="color: #f87171; font-size: 14px;">⚠️ Location not found. Try typing another city or location.</p>`;
            }
          }
        })
        .catch(err => {
          console.error("Geocoding error:", err);
          if (hospitalList) {
            hospitalList.innerHTML = `<p style="color: #f87171; font-size: 14px;">⚠️ Error finding location. Please try again.</p>`;
          }
        });
    };
    
    searchBtn.addEventListener("click", performSearch);
    manualInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        performSearch();
      }
    });
  }

  const fallbackToIpLocation = () => {
    console.log("[Location] GPS failed or blocked. Trying IP Geolocation fallback...");
    fetch('https://ipapi.co/json/')
      .then(res => res.json())
      .then(data => {
        if (data && data.latitude && data.longitude) {
          console.log("[Location] IP Geolocation resolved coordinates:", data.city, data.latitude, data.longitude);
          showVetHospitals(data.latitude, data.longitude);
        } else {
          // Ultimate fallback (New Delhi)
          showVetHospitals(28.6139, 77.2090);
        }
      })
      .catch(err => {
        console.error("[Location] IP Geolocation failed:", err);
        // Ultimate fallback (New Delhi)
        showVetHospitals(28.6139, 77.2090);
      });
  };

  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (position) => {
        console.log("[Location] GPS resolved coordinates:", position.coords.latitude, position.coords.longitude);
        showVetHospitals(position.coords.latitude, position.coords.longitude);
      },
      () => {
        fallbackToIpLocation();
      },
      { timeout: 8000 }
    );
  } else {
    fallbackToIpLocation();
  }
}

let leafletMap = null;

function showVetHospitals(lat, lng) {
  const mapElement = document.getElementById("vetMap");
  const hospitalList = document.getElementById("vetHospitalList");
  if (!mapElement) return;

  // Destroy existing map if re-initialized
  if (leafletMap) {
    leafletMap.remove();
    leafletMap = null;
  }

  // Create dark-themed Leaflet map
  leafletMap = L.map(mapElement).setView([lat, lng], 13);
  
  // Invalidate size immediately and after a short timeout to handle the container transition/display rendering
  leafletMap.invalidateSize();
  setTimeout(() => {
    if (leafletMap) leafletMap.invalidateSize();
  }, 200);

  // Dark tile layer (CartoDB Dark Matter)
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap &copy; CartoDB',
    maxZoom: 19
  }).addTo(leafletMap);

  // User location marker (blue circle)
  L.circleMarker([lat, lng], {
    radius: 10,
    color: '#3b82f6',
    fillColor: '#60a5fa',
    fillOpacity: 0.9,
    weight: 3
  }).addTo(leafletMap).bindPopup("<strong style='color:#1e293b;'>📍 You are here</strong>");

  // Query Overpass API for veterinary hospitals within ~8km radius
  const overpassQuery = `
    [out:json][timeout:10];
    (
      node["amenity"="veterinary"](around:8000,${lat},${lng});
      way["amenity"="veterinary"](around:8000,${lat},${lng});
      node["healthcare"="veterinary"](around:8000,${lat},${lng});
      node["shop"="pet"](around:8000,${lat},${lng});
    );
    out center 10;
  `;

  const overpassUrl = `https://overpass-api.de/api/interpreter?data=${encodeURIComponent(overpassQuery)}`;

  if (hospitalList) {
    hospitalList.innerHTML = `<p style="color: #94a3b8; font-size: 14px;">🔍 Searching nearby veterinary clinics...</p>`;
  }

  fetch(overpassUrl)
    .then(res => res.json())
    .then(data => {
      const places = data.elements || [];

      if (places.length === 0 && hospitalList) {
        hospitalList.innerHTML = `<p style="color: #f59e0b; font-size: 14px;">No veterinary clinics found within 8km. Try searching manually in your area.</p>`;
        return;
      }

      if (hospitalList) hospitalList.innerHTML = "";

      const vetIcon = L.divIcon({
        html: '<div style="background:#ef4444; color:white; border-radius:50%; width:28px; height:28px; display:flex; align-items:center; justify-content:center; font-size:14px; border:2px solid white; box-shadow:0 2px 6px rgba(0,0,0,0.4);">🏥</div>',
        iconSize: [28, 28],
        iconAnchor: [14, 14],
        className: ''
      });

      places.slice(0, 6).forEach((place) => {
        const pLat = place.lat || (place.center && place.center.lat);
        const pLng = place.lon || (place.center && place.center.lon);
        if (!pLat || !pLng) return;

        const name = place.tags?.name || "Veterinary Clinic";
        const address = place.tags?.["addr:street"] || place.tags?.["addr:full"] || "Address not available";
        const phone = place.tags?.phone || "";

        // Add marker on map
        const marker = L.marker([pLat, pLng], { icon: vetIcon }).addTo(leafletMap);
        marker.bindPopup(`<strong style="color:#1e293b;">${name}</strong><br><span style="color:#475569;">${address}</span>${phone ? '<br>📞 ' + phone : ''}`);

        // Distance calculation
        const dist = getDistanceKm(lat, lng, pLat, pLng);
        const distText = dist < 1 ? `${Math.round(dist * 1000)}m away` : `${dist.toFixed(1)} km away`;

        // Add to hospital list card
        if (hospitalList) {
          hospitalList.innerHTML += `
            <div style="background: rgba(30, 41, 59, 0.6); padding: 14px; border-radius: 10px; border: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; cursor: pointer; transition: background 0.2s;" 
                 onmouseover="this.style.background='rgba(30, 41, 59, 0.9)'" 
                 onmouseout="this.style.background='rgba(30, 41, 59, 0.6)'"
                 onclick="if(leafletMap) { leafletMap.setView([${pLat}, ${pLng}], 16); leafletMap.invalidateSize(); }">
              <div>
                <h4 style="color: white; font-size: 15px; margin: 0;">🏥 ${name}</h4>
                <p style="color: #94a3b8; font-size: 13px; margin: 4px 0 0 0;">${address}</p>
                ${phone ? `<p style="color: #60a5fa; font-size: 12px; margin: 3px 0 0 0;">📞 ${phone}</p>` : ''}
              </div>
              <span style="background: #1e293b; padding: 5px 10px; border-radius: 8px; font-size: 12px; color: #f59e0b; white-space: nowrap;">${distText}</span>
            </div>
          `;
        }
      });

      // Fit map bounds to show all markers with correct leaflet container dimensions
      if (places.length > 0) {
        leafletMap.invalidateSize();
        const bounds = L.latLngBounds([[lat, lng]]);
        places.slice(0, 6).forEach(p => {
          const pLat = p.lat || (p.center && p.center.lat);
          const pLng = p.lon || (p.center && p.center.lon);
          if (pLat && pLng) bounds.extend([pLat, pLng]);
        });
        setTimeout(() => {
          if (leafletMap) {
            leafletMap.invalidateSize();
            leafletMap.fitBounds(bounds, { padding: [40, 40] });
          }
        }, 150);
      }
    })
    .catch(err => {
      console.error("Overpass API error:", err);
      if (hospitalList) {
        hospitalList.innerHTML = `<p style="color: #f87171; font-size: 14px;">⚠️ Failed to search for nearby clinics. Please try again.</p>`;
      }
    });
}

// Haversine distance in km
function getDistanceKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon/2) * Math.sin(dLon/2);
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

// --- 5. Speech Recognition (Speech to Text) and Text-to-Speech ---

function setupSpeechControls(btnId, selectId, statusId, textareaId) {
  const btn = document.getElementById(btnId);
  const select = document.getElementById(selectId);
  const status = document.getElementById(statusId);
  const textarea = document.getElementById(textareaId);

  if (!btn || !select || !status || !textarea) return;

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    btn.style.display = "none";
    status.innerText = "⚠️ Speech input not supported in this browser";
    return;
  }

  let recognition = null;
  let isListening = false;

  btn.addEventListener("click", () => {
    if (isListening) {
      if (recognition) recognition.stop();
      return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = select.value;

    recognition.onstart = () => {
      isListening = true;
      btn.innerHTML = "🛑 Stop";
      btn.style.background = "#ef4444";
      status.innerText = "🎙️ Listening... Speak now";
    };

    recognition.onend = () => {
      isListening = false;
      btn.innerHTML = "🎤 Speak";
      btn.style.background = "linear-gradient(to right,#2563eb,#7c3aed)";
      status.innerText = "";
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      textarea.value = textarea.value.trim() 
        ? textarea.value + " " + transcript 
        : transcript;
      status.innerText = "✅ Transcribed!";
      setTimeout(() => { status.innerText = ""; }, 2000);
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
      if (event.error === 'not-allowed') {
        status.innerText = "⚠️ Mic permission denied";
      } else {
        status.innerText = "⚠️ Error: " + event.error;
      }
      setTimeout(() => { status.innerText = ""; }, 3000);
    };

    recognition.start();
  });
}

function speakText(text, lang) {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;

    // Pick the best matching voice for the selected language
    const voices = window.speechSynthesis.getVoices();
    const langBase = lang.split('-')[0]; // e.g. 'hi' from 'hi-IN'
    const exactMatch = voices.find(v => v.lang === lang);
    const partialMatch = voices.find(v => v.lang.startsWith(langBase));
    if (exactMatch) {
      utterance.voice = exactMatch;
    } else if (partialMatch) {
      utterance.voice = partialMatch;
    }

    utterance.rate = 0.9;
    window.speechSynthesis.speak(utterance);
  } else {
    alert("Speech synthesis is not supported on this browser.");
  }
}

// Preload voices (Chrome loads them async)
if ('speechSynthesis' in window) {
  window.speechSynthesis.onvoiceschanged = () => { window.speechSynthesis.getVoices(); };
  window.speechSynthesis.getVoices();
}

// Bind plant screen voice elements
setupSpeechControls("plantVoiceBtn", "plantVoiceLang", "plantVoiceStatus", "plantDescription");
// Bind animal screen voice elements
setupSpeechControls("animalVoiceBtn", "animalVoiceLang", "animalVoiceStatus", "animalDescription");

// Bind read-aloud buttons
const plantReadAloudBtn = document.getElementById("plantReadAloudBtn");
if (plantReadAloudBtn) {
  plantReadAloudBtn.addEventListener("click", () => {
    const title = document.getElementById("resultCropDisease").innerText;
    const desc = document.getElementById("resultDescription").innerText;
    const recs = Array.from(document.getElementById("resultRecommendations").children).map(li => li.innerText).join(". ");
    const fullText = `${title}. Diagnosis: ${desc}. Recommended solutions: ${recs}`;
    const lang = document.getElementById("plantVoiceLang").value;
    speakText(fullText, lang);
  });
}

const animalReadAloudBtn = document.getElementById("animalReadAloudBtn");
if (animalReadAloudBtn) {
  animalReadAloudBtn.addEventListener("click", () => {
    const title = document.getElementById("animalResultTitle").innerText;
    const desc = document.getElementById("animalResultDescription").innerText;
    const recs = Array.from(document.getElementById("animalResultRecommendations").children).map(li => li.innerText).join(". ");
    const fullText = `${title}. Diagnosis: ${desc}. Recommended solutions: ${recs}`;
    const lang = document.getElementById("animalVoiceLang").value;
    speakText(fullText, lang);
  });
}

// Load and save settings helpers
function initSettingsPage() {
  const hfApiTokenInput = document.getElementById("hfApiTokenInput");
  const modelIdInput = document.getElementById("modelIdInput");
  const saveSettingsBtn = document.getElementById("saveSettingsBtn");
  const settingsSaveStatus = document.getElementById("settingsSaveStatus");
  const pipelineModeSelect = document.getElementById("pipelineModeSelect");
  const vitModelIdInput = document.getElementById("vitModelIdInput");
  const llamaModelIdInput = document.getElementById("llamaModelIdInput");

  if (!hfApiTokenInput || !modelIdInput || !saveSettingsBtn) return;

  // Load existing settings
  fetch('/api/settings')
    .then(res => res.json())
    .then(data => {
      if (data.api_token) hfApiTokenInput.value = data.api_token;
      if (data.model_id) modelIdInput.value = data.model_id;
      if (data.pipeline_mode && pipelineModeSelect) {
        pipelineModeSelect.value = data.pipeline_mode;
      }
      if (data.vit_model_id && vitModelIdInput) vitModelIdInput.value = data.vit_model_id;
      if (data.llama_model_id && llamaModelIdInput) llamaModelIdInput.value = data.llama_model_id;
    })
    .catch(err => console.error("Error loading settings:", err));

  // Save settings
  saveSettingsBtn.addEventListener("click", () => {
    const payload = {
      api_token: hfApiTokenInput.value,
      model_id: modelIdInput.value,
      inference_mode: "api",
      pipeline_mode: pipelineModeSelect ? pipelineModeSelect.value : "multi",
      vit_model_id: vitModelIdInput ? vitModelIdInput.value : "wambugu71/crop_leaf_diseases_vit",
      llama_model_id: llamaModelIdInput ? llamaModelIdInput.value : "meta-llama/Llama-3.1-8B-Instruct"
    };

    fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
      .then(res => res.json())
      .then(() => {
        settingsSaveStatus.style.display = "block";
        setTimeout(() => { settingsSaveStatus.style.display = "none"; }, 3000);
      })
      .catch(err => {
        console.error("Error saving settings:", err);
        alert("Failed to save settings.");
      });
  });
}

// Initialize Dashboard and Analytics Chart on first load
document.addEventListener("DOMContentLoaded", () => {
  // Sync light/dark theme preference from localStorage immediately
  const darkModeBtn = document.getElementById("darkModeBtn");
  const savedTheme = localStorage.getItem("theme") || "dark";
  
  if (savedTheme === "light") {
    document.body.classList.add("light-theme");
    if (darkModeBtn) darkModeBtn.innerText = "☀️ Light Mode";
  } else {
    document.body.classList.remove("light-theme");
    if (darkModeBtn) darkModeBtn.innerText = "🌙 Dark Mode";
  }

  if (darkModeBtn) {
    darkModeBtn.addEventListener("click", () => {
      document.body.classList.toggle("light-theme");
      const isLight = document.body.classList.contains("light-theme");
      if (isLight) {
        localStorage.setItem("theme", "light");
        darkModeBtn.innerText = "☀️ Light Mode";
      } else {
        localStorage.setItem("theme", "dark");
        darkModeBtn.innerText = "🌙 Dark Mode";
      }
    });
  }

  loadDashboardStats();
  loadAnalyticsData();
  initSettingsPage();
});