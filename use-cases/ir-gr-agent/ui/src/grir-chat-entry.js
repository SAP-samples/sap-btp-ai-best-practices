import "./pages/grir-chat/grir-chat.css";
import html from "./pages/grir-chat/grir-chat.html?raw";
import initGrirChatPage from "./pages/grir-chat/grir-chat.js";

document.getElementById("grir-app").innerHTML = html;
initGrirChatPage();
