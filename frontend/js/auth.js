let isLoginMode = true;

function openAuthModal() {
  document.getElementById("authModal").style.display = "flex";
  document.getElementById("authError").style.display = "none";
}

function closeAuthModal() {
  document.getElementById("authModal").style.display = "none";
}

function toggleAuthMode() {
  isLoginMode = !isLoginMode;
  document.getElementById("authTitle").textContent = isLoginMode ? "Login" : "Sign Up";
  document.getElementById("authToggleText").textContent = isLoginMode ? "Need an account? Sign up" : "Already have an account? Login";
  document.getElementById("authError").style.display = "none";
}

async function submitAuth() {
  const user = document.getElementById("authUsername").value.trim();
  const pass = document.getElementById("authPassword").value.trim();
  const errorEl = document.getElementById("authError");
  
  if (!user || !pass) {
    errorEl.textContent = "Please fill in all fields.";
    errorEl.style.display = "block";
    return;
  }
  
  const endpoint = isLoginMode ? "/api/auth/login" : "/api/auth/register";
  
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: user, password: pass })
    });
    const data = await res.json();
    
    if (res.ok) {
      localStorage.setItem("token", data.token);
      localStorage.setItem("username", data.username);
      closeAuthModal();
      checkAuthState();
    } else {
      errorEl.textContent = data.message || "An error occurred.";
      errorEl.style.display = "block";
    }
  } catch (err) {
    errorEl.textContent = "Unable to connect to server.";
    errorEl.style.display = "block";
  }
}

function logout() {
  localStorage.removeItem("token");
  localStorage.removeItem("username");
  checkAuthState();
}

function checkAuthState() {
  const token = localStorage.getItem("token");
  const username = localStorage.getItem("username");
  const authHeaders = document.querySelectorAll('.auth-header-container'); // Need to map to header injection
  
  let content = "";
  if (token && username) {
    content = `
      <span style="font-size:0.9rem; font-weight:600; color:#6c63ff; margin-right:1rem;">Hi, ${username}!</span>
      <button class="btn-secondary btn-sm" onclick="openFavModal()">My Favorites</button>
      <button class="btn-sm" style="background:transparent; border:none; color:#888; cursor:pointer; text-decoration:underline; font-size:0.8rem; margin-left:0.5rem;" onclick="logout()">Logout</button>
    `;
  } else {
    content = `<button class="btn-secondary btn-sm" onclick="openAuthModal()">Login / Signup</button>`;
  }
  
  authHeaders.forEach(el => { el.innerHTML = content; });
}

// ─── Favorites ────────────────────────────────────────────────────────
async function loadFavorites() {
  const token = localStorage.getItem("token");
  if (!token) return [];
  
  try {
    const res = await fetch("/api/favorites", {
      headers: { "Authorization": "Bearer " + token }
    });
    if (res.ok) {
      const data = await res.json();
      return data.favorites;
    }
  } catch(e) {}
  return [];
}

async function toggleFavorite(btnElement, hairstyleName) {
  const token = localStorage.getItem("token");
  if (!token) {
    openAuthModal();
    return;
  }
  
  const isActive = btnElement.classList.contains("active");
  const endpoint = isActive ? "/api/favorites/remove" : "/api/favorites/add";
  
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        "Authorization": "Bearer " + token
      },
      body: JSON.stringify({ hairstyle_name: hairstyleName })
    });
    
    if (res.ok) {
      if (isActive) btnElement.classList.remove("active");
      else btnElement.classList.add("active");
    }
  } catch (err) {
    console.error(err);
  }
}

async function openFavModal() {
  const favList = document.getElementById("favList");
  favList.innerHTML = "<p>Loading...</p>";
  document.getElementById("favModal").style.display = "flex";
  
  const favs = await loadFavorites();
  if (favs.length === 0) {
    favList.innerHTML = "<p style='color:#888; text-align:center;'>You haven't saved any hairstyles yet.</p>";
  } else {
    favList.innerHTML = favs.map(f => `
      <div class="fav-item">
        <strong>${f.hairstyle_name}</strong>
        <button class="btn-sm" style="background:#fff0f0; color:#c0392b; border:1px solid #f5c6cb" onclick="removeFavoriteFromModal('${f.hairstyle_name}')">Remove</button>
      </div>
    `).join("");
  }
}

function closeFavModal() {
  document.getElementById("favModal").style.display = "none";
}

async function removeFavoriteFromModal(name) {
  const token = localStorage.getItem("token");
  if (!token) return;
  await fetch("/api/favorites/remove", {
    method: "POST",
    headers: { 
      "Content-Type": "application/json",
      "Authorization": "Bearer " + token
    },
    body: JSON.stringify({ hairstyle_name: name })
  });
  openFavModal(); // Refresh
}

// ─── Init ─────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", checkAuthState);
