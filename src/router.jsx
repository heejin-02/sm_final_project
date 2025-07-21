// src/router.jsx
import Home from "./pages/Home";
import Detail from "./pages/Detail";
import Write, { handleSubmit } from "./pages/Write";

export function router() {
  const app = document.querySelector("#app");
  if (!app) {
    throw new Error("App element not found");
  }

  const getLayout = (title, page) => `
    <div>
      <header class="bg-[#007AFF] text-white flex w-full justify-between items-center px-[20px] py-[12px]">
        <h1 class="text-[28px]">${title}</h1>
        ${page === "detail" ? `<button class="text-lg" onclick="location.href='#/'">홈으로</button>` : ""}
        ${page === "write"  ? `<button id="submit-button" class="text-lg">완료</button>`      : ""}
      </header>
      <main class="p-4"></main>
    </div>
  `;

  const hash = window.location.hash.split("?")[0];
  // 홈
  if (hash === "" || hash === "#/") {
    app.innerHTML = getLayout("Diary", "home");
    Home(app);

  // 쓰기
  } else if (hash === "#/write") {
    app.innerHTML = getLayout("쓰기", "write");
    Write(app);
    const submitButton = app.querySelector("#submit-button");
    if (submitButton) {
      submitButton.addEventListener("click", handleSubmit);
    }

  // 상세
  } else if (hash === "#/detail") {
    app.innerHTML = getLayout("읽기", "detail");
    Detail(app);

  // 404
  } else {
    app.innerHTML = `
      <h1>404 Not Found</h1>
      <p>해당 페이지를 찾을 수 없습니다.</p>
    `;
    console.warn("Unknown path, redirecting to home");
  }
}
