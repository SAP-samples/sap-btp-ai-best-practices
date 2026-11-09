/**
 * Cookie Statement page.
 * Renders the SAP cookie statement adapted for AI4U Ideation Agent.
 */
export default function initCookiesPage() {
  const root = document.getElementById("cookies-root");
  if (!root) return;

  root.innerHTML = _renderCookiesHTML();

  const backBtn = root.querySelector("#cookies-back-btn");
  if (backBtn) {
    backBtn.addEventListener("click", (e) => {
      e.preventDefault();
      if (window.history.length > 1) {
        window.history.back();
      } else {
        window.location.href = "/";
      }
    });
  }
}

function _renderCookiesHTML() {
  return `
    <div class="privacy-page">
      <a class="privacy-page__back" href="/" id="cookies-back-btn">← Back</a>

      <h1>Cookie Statement</h1>

      <p>This Cookie Statement was updated on 2026-03-06.</p>

      <p>This Cookie Statement describes how SAP (hereinafter also "We", "Our") uses cookies and similar technologies to collect and store information when you visit the AI4U Ideation Agent. SAP's Privacy Statement applies in addition to this Cookie Statement. SAP's Privacy Statement informs you about the way SAP uses, stores and protects personal data collected. It also informs you of your data protection rights and how to exercise them. We recommend that you read <a href="/privacy">SAP's Privacy Statement</a>.</p>

      <h2>What are Cookies and similar technologies?</h2>

      <p>Cookies are small files placed on your device (computer, tablet or smartphone). When you access a website, a cookie is placed on your device and it will send information to the party that placed the cookie.</p>

      <p>There are other technologies performing a similar function to cookies. These include inter alia web beacons, clear gifs, and social plug-ins. These are often used in conjunction with cookies to help a website owner understand its users better.</p>

      <h2>What are first party Cookies?</h2>

      <p>SAP's websites contain first party Cookies. First party cookies are cookies that are specific to the website that created them. These cookies enable SAP to operate an efficient service and to track patterns of user behavior to SAP's website.</p>

      <h2>What are third party Cookies?</h2>

      <p>The difference between a first party cookie and a third party cookie relates to who places the cookie on your device. SAP sometimes allows third parties to place Cookies on your device. Third party cookies are placed on your device by a third party (i.e., not by SAP). While SAP allows third parties to access SAP's website to place a third-party cookie on your device, SAP does not retain control over the information supplied by the cookies, nor does SAP retain access to this information. This information is controlled wholly by that third party according to the respective privacy policy of the third party. For more information about these Cookies, please click the "Cookie Preferences" link in the footer of the webpage.</p>

      <h2>Analytics and Digital Marketing in the United States of America</h2>

      <p>In the United States SAP routinely works with third party analytics and digital marketing providers who facilitate and support SAP (and partner) marketing activities including targeted communications. These activities may involve the sharing of limited Personal Information (personal identifiers and contact information, your interests and interactions with SAP, professional/employment information) for those specific business purposes. In addition to the above, SAP may also share your personal data with other third parties, or for other business purposes, with your explicit consent.</p>

      <p>SAP endeavors to recognize common opt-out preference signals at a browser level, and frictionlessly exclude you from marketing cookies and associated tracking.</p>

      <h2>What cookies are used?</h2>

      <p>SAP wants you to be in a position to make an informed decision for or against the use of cookies which are not strictly necessary for technical features on the AI4U Ideation Agent. If you elect to reject cookies used for advertising, you will be shown advertising that is less targeted to your interests. This will still allow you to use all of the functionality of the AI4U Ideation Agent.</p>

      <p>When tracking and evaluating the usage behavior of users of the Web Presence by means of cookies or similar technologies includes the processing of your personal data, SAP conducts the processing based on the following legal permissions:</p>

      <ul>
        <li>GDPR Article 6.I (a) if it is necessary that We ask you for your consent to process your personal data,</li>
        <li>GDPR Article 6.I (b) if necessary to fulfill (pre-)contractual obligations with you,</li>
        <li>GDPR Article 6.I (f) if necessary to fulfill (pre-)contractual obligations with the company or other legal body you represent as a customer contact (legitimate interest to efficiently perform or manage SAP's business operation),</li>
        <li>or equivalent legal permissions under other relevant national laws, when applicable.</li>
      </ul>

      <p>SAP differentiates between Required Cookies that are absolutely necessary to enable technical core functionalities, Functional Cookies that allow SAP to analyze the site usage, and Advertising Cookies that are used to serve ads relevant to your interests.</p>

      <h3>Required Cookies</h3>

      <table>
        <tr>
          <th>Name</th>
          <th>Purpose</th>
          <th>Lifespan</th>
        </tr>
        <tr>
          <td>glt_*</td>
          <td>SAP Customer Data Cloud (Gigya) login token that maintains the authenticated user session</td>
          <td>Session</td>
        </tr>
        <tr>
          <td>gig_bootstrap_*</td>
          <td>SAP Customer Data Cloud (Gigya) SDK bootstrap data required for SDK initialization</td>
          <td>Session</td>
        </tr>
        <tr>
          <td>gig_canary_*</td>
          <td>SAP Customer Data Cloud (Gigya) SDK version tracking for gradual rollout</td>
          <td>Session</td>
        </tr>
        <tr>
          <td>gig_canary_ver</td>
          <td>SAP Customer Data Cloud (Gigya) SDK canary version identifier</td>
          <td>Session</td>
        </tr>
        <tr>
          <td>gmid</td>
          <td>SAP Customer Data Cloud (Gigya) unique device identifier used to associate sessions with a device</td>
          <td>1 year</td>
        </tr>
        <tr>
          <td>ucid</td>
          <td>SAP Customer Data Cloud (Gigya) unique client identifier used for analytics and session correlation</td>
          <td>1 year</td>
        </tr>
        <tr>
          <td>hasGmid</td>
          <td>SAP Customer Data Cloud (Gigya) flag indicating whether a device identifier has been assigned</td>
          <td>1 year</td>
        </tr>
      </table>

      <h2>Local Storage</h2>

      <p>In addition to cookies, this application uses browser local storage to persist your session data and preferences. This data remains on your device and is not transmitted to our servers unless explicitly required for application functionality.</p>

      <table>
        <tr>
          <th>Name</th>
          <th>Purpose</th>
          <th>Lifespan</th>
        </tr>
        <tr>
          <td>ai4u_auth_user</td>
          <td>Stores the authenticated user's session information (name, email, auth provider)</td>
          <td>Persistent</td>
        </tr>
        <tr>
          <td>ai4u_auth_token</td>
          <td>Stores the authentication token for API requests</td>
          <td>Persistent</td>
        </tr>
      </table>

      <h2>How can you manage and delete Cookies?</h2>

      <p>SAP provides you with the option to adjust your preferences for Functional and Advertising Cookies when such cookies are placed on your device. In such a case, you can access preferences at any time by clicking on the "Cookie Preferences" link in the footer of the AI4U Ideation Agent.</p>

      <p>You can also block and delete cookies by changing your browser settings. To manage cookies using your browser settings, most browsers allow you to refuse or accept all cookies or only to accept certain types of cookies. The process for the management and deletion of cookies can be found in the help function integrated in your browser. If you wish to limit the use of cookies, you may not be able to use all the interactive functions.</p>

      <h2>What are Media Plug-Ins?</h2>

      <p>A Media plug-in embedded on a website allows you to watch remote content from a third party media provider's website. You can recognize media plug-ins as views displayed on website pages; e.g., for YouTube, you can recognize the media plugin by a video showing the play button.</p>

      <h3>How do media plug-ins on SAP websites work?</h3>

      <p>If you want to use one of content of these media plug-ins, click on the play-symbol within the media plug-in to establish a direct connection to the server of the respective media provider. The stream of the media will be redirected from the media provider to your browser.</p>

      <p>If you have a user account on the media provider and are logged in at the time you activate the media plug-in, the media provider can associate your visit to SAP's websites with your user account. In this situation, data transmissions can also take place that are initiated and controlled by the respective media provider. Your connection to a media provider, the data transfers taking place between the network and your system, and your interactions on that platform are governed solely by the privacy and cookie statement or policy of that media provider. Personal Data may also reach providers in countries outside the European Economic Area that may not guarantee an "adequate level of protection" for the processing of Personal Data in accordance with EU standards. If you do not want to share any data with the media provider in connection with SAP's website, do not click on the media plug-in on SAP's website.</p>

      <h3>On which legal basis does SAP make media plug-ins available?</h3>

      <p>SAP uses media plug-ins on its websites on the basis of its legitimate interest to make remote content visible to the user according to Article 6 (1) lit. f GDPR or the applicable national law.</p>
    </div>
  `;
}
