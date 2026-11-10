/**
 * Privacy Statement page.
 * Renders the SAP privacy content adapted for AI4U Ideation Agent.
 */
export default function initPrivacyPage() {
  const root = document.getElementById("privacy-root");
  if (!root) return;

  root.innerHTML = _renderPrivacyHTML();

  const backBtn = root.querySelector("#privacy-back-btn");
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

function _renderPrivacyHTML() {
  return `
    <div class="privacy-page">
      <a class="privacy-page__back" href="/" id="privacy-back-btn">← Back</a>

      <h1>Privacy Statement</h1>

      <p>This Privacy Statement was updated on 2026-03-06. We have created this Privacy Statement to demonstrate the firm commitment of SAP (hereinafter "We", "SAP", "Us" or "Our") to the individual's right to data protection and privacy. It outlines how SAP processes information that can be used to directly or indirectly identify an individual (hereinafter "Personal Data"). Processing in the context of this Privacy Statement means any collection, use, transmission, disclosure, erasure or any other similar operation based on Personal Data (hereinafter "Processing" or "Process").</p>

      <p>SAP is processing information including Personal Data about the users of AI4U Ideation Agent website using cookies or similar technologies for the purposes set out in the <a href="/cookies" target="_blank" rel="noopener">Cookie Statement</a>.</p>

      <h2>A. GENERAL INFORMATION</h2>

      <h3>Who is the responsible SAP entity</h3>

      <p>This Privacy Statement applies to the collection and processing of personal data:</p>

      <ul>
        <li>during the central operation of this website and other globally operated SAP business activities by
          <ul>
            <li><strong>SAP SE</strong>, Dietmar-Hopp-Allee 16 Walldorf 69190, Germany if you are in a member state of the EU or the EWR or in any of the countries of Andorra, Faroe Islands, Guernsey, Isle of Man, Jersey, Switzerland, or the United Kingdom or by</li>
            <li><strong>SAP America Inc.</strong>, 3809 West Chester Pike, Suite 200, Newtown Square, PA 19073, USA if you are in any other country, or</li>
            <li>a <strong>specific SAP group entity</strong> as may be stated in the Additional Country and Regional Specific Provisions at the end of this privacy statement.</li>
          </ul>
        </li>
        <li>in the context of a pre-contractual or contractual business relationship with you or your employer by a <strong>local SAP group entity</strong></li>
        <li>in the context of a registration form when a <strong>specific SAP group entity</strong> is directly collecting personal data for the purpose of registering to a service or event and is therefore presented as the relevant controller on this registration page or website by referencing to this privacy statement. Where a registration form is presented on this website, the controller may vary depending on the actual offering or the purpose of the data collection, but it is in any case displayed on the individual registration form's privacy statement.</li>
      </ul>

      <p>You can reach SAP Group's data protection officer any time at privacy[@]sap.com.</p>

      <h4>SAP's Web Services</h4>

      <p>SAP processes your Personal Data to operate web presences, web offerings, or online events ("Web Services")</p>

      <ul>
        <li>to provide the Web Services and functions, create and administer your online account, updating, securing, troubleshooting the service, providing support, improving, and developing the Web Services, answering and fulfilling your requests or instructions.</li>
        <li>to manage and ensure the security of Our Web Services and prevent and detect security threats, fraud or other criminal or malicious activities and as reasonably necessary to enforce the Web Services terms, to establish or preserve a legal claim or defense, to prevent fraud or other illegal activities, including attacks on Our information technology systems.</li>
        <li>to create specific user profiles that may be specific to a single Web Service of SAP, but also allows you to access SAP's other Web Services. It is your choice whether or not to use any of these additional Web Service. If you do, SAP will make your Personal Data available to such other Web Service to provide you with initial access. Kindly note that without your consent for SAP to create such user profiles, SAP will not be able to offer such services to you where your consent is a statutory requirement that SAP can provide these services to you.</li>
        <li>to process information that relates to your visit to Our Web Services to improve your user experience, identify your individual demand and to personalize the way We provide you with the information you are looking for. For this purpose, We collect information regardless of whether you register with a user profile or not.</li>
        <li>to create your user profile for the AI4U Ideation Agent. Through the user profile you can share Personal Data about you with other users, such as your name, photo, social media accounts, postal or email address, telephone number, personal interests, skills, and basic information about your company. The user profiles serve to personalize the interactions between the users (for example, by way of messaging or follow functionality) and to allow SAP to foster the collaboration and quality of communication through such offerings. The profile settings of the relevant Web Services allow you to determine which information you want to share.</li>
        <li>to share basic participant information (your name, company, and email address) with other participants of the same event, seminar, or webinar to promote the interaction between the participants and stimulate the communication and the exchange of ideas.</li>
      </ul>

      <p>When operating SAP's Web Services, SAP processes your Personal Data if and to the extent,</p>

      <ul>
        <li>SAP obtained your consent, if required by law, to process your Personal Data for this purpose,</li>
        <li>necessary to fulfill (pre-)contractual obligations with you,</li>
        <li>necessary to fulfill legal requirements applicable to SAP,</li>
        <li>necessary to pursue SAP's legitimate interest to efficiently perform or manage SAP's Web Services and business operation and assert or defend itself against legal claims. We believe that SAP's interest in pursuing these business purposes is legitimate and thereby not outweighed by your personal rights and interest to refrain processing for such purpose. In any of these cases, We duly factor into Our balancing test: the business purpose reasonably pursued by SAP in the given case, the categories, amount and sensitivity of Personal Data that is necessarily being processed, the level of protection of your Personal Data which is ensured by means of Our general data protection policies, guidelines, and processes, and the rights you have in relation to the processing activity.</li>
      </ul>

      <h3>For what purposes does SAP process your Personal Data and based on what legal basis?</h3>

      <p>Depending on the applicable law, the Processing of Personal Data is subject to a justification, sometimes referred to as legal basis.</p>

      <h4>SAP's compliance with statutory obligations</h4>

      <ul>
        <li>SAP processes your Personal Data for the purpose of ensuring an adequate level of technical and organizational security of SAP's products, services, online events, facilities, and premises. For this, SAP will take the measures necessary to verify or maintain the quality and safety of a product or service which is owned, manufactured by or for, or controlled by SAP. This may comprise the use of Personal Data for sufficient identification and authorization of designated users, internal quality control through auditing, analysis, and research, debugging to identify and repair errors that impair existing or intended functionality, account and network security, replication for loss prevention, detecting security incidents, protection against malicious, deceptive, fraudulent, or illegal activity, and prosecuting those responsible for such kind of activity. We may further process your name, likeness, and other contact or compliance related data when you visit a local SAP affiliate or lab in the context of access management and video surveillance to protect the security and safety of Our locations and assets.</li>
        <li>SAP processes Personal Data (name, surname, country, IP address) to the extent necessary to fulfil sanctions and embargo requirements under European Economic Area ("EEA") laws to which SAP is subject, and laws and regulations extraterritorial to the EEA (based on SAP's legitimate interest).</li>
        <li>If necessary, SAP uses Personal Data to prevent or prosecute criminal activities such as any form of cybercrime, the illegal use of Our products and services or fraud, to assert Our rights or defend SAP against legal claims.</li>
        <li>To comply with data protection and privacy laws (e.g. to respond to your data subject request) and unfair competition laws. Depending on the country in which the relevant SAP Group company operates, SAP may process Personal Data necessary to accommodate your data protection and privacy choices for the receipt of such information and, when necessary to ensure compliance, exchange such information with the other entities of the SAP Group.</li>
      </ul>

      <p>When ensuring compliance, SAP processes your Personal Data if and to the extent necessary to fulfill legal requirements under European Union or EU Member State law to which SAP is subject, and laws and regulations extraterritorial to the EU (legitimate interest to comply with extraterritorial laws and regulations).</p>

      <h4>SAP's business relationships</h4>

      <p>SAP processes Personal Data to pursue its business relationships with customers, partners, and others to fulfill pre-contractual and contractual business relations. This may include satisfying requests, processing orders, delivering an ordered product or service, or engaging in any other relevant action to establish, fulfill and maintain Our business relationships.</p>

      <ul>
        <li>Products and services may include any of SAP's on-prem and cloud software products, web services, apps, online-forums, webinars and events, non-marketing related newsletters, white papers, tutorials, trainings, as well as other offerings like contests or sweepstakes. When you purchase or intend to purchase products or services from SAP on behalf of a corporate customer or are otherwise associated as contact person for the business relationship between SAP and a corporate customer or partner ("Customer Contact"), SAP will use your Personal Data for this purpose.</li>
        <li>Customer Satisfaction: Within an existing business relationship between you or your employer and SAP, SAP processes your Personal Data to help Us understand how satisfied you are with the functionality and quality of Our products and services, to provide you with relevant information on Our latest product announcements, software updates or upgrades, events, special offers, and other information about SAP's software and services that is relevant and useful to you.</li>
        <li>To keep you up to date: Within an existing business relationship between you or your employer and SAP, SAP processes your Personal Data to inform you about SAP's products or services which are similar or relate to products and services you or your employer have already purchased or used. SAP will inform you by email or phone about such news only as far as it is allowed by law, or if SAP has collected such information in the context of the business relationship. You are entitled to object to SAP's use for this purpose at any time by selecting the opt-out option at the bottom of each marketing related approach.</li>
        <li>Feedback requests and surveys: To the extent allowed by applicable law, SAP may contact you for feedback regarding the improvement of the relevant material, product, or service. SAP may also invite you to participate in questionnaires and surveys. These will generally be designed so you can participate without having to provide information that identifies you as a participant. If you nonetheless provide your Personal Data, SAP will use it for the purpose stated in the questionnaire or survey or to improve its products and services.</li>
        <li>Personalized Content: SAP processes information about your interactions with SAP across its various business areas and its offerings to provide you with the requested products and services and to improve Our personal communications with you.</li>
        <li>Advertising ID's: SAP may create a hashed user ID to provide to third party operated social networks or other web offerings (such as Twitter, LinkedIn, Facebook, Instagram or Google). This information is then matched against the third party's own user database to display to you more relevant SAP content.</li>
      </ul>

      <h3>What categories of Personal Data does SAP process?</h3>

      <p>SAP processes the following categories of Personal Data: Identification Data, Contact Information, Employment and Professional Data, System Access and Usage Data, Behavioral and Preference Data.</p>

      <p>If SAP processes special categories of Personal Data under applicable law, SAP will ask you for your consent in a specific declaration.</p>

      <h3>From What Types of Third Parties does SAP obtain Personal Data?</h3>

      <p>SAP generally aims to collect Personal Data directly from you. If you are obliged by statutory law or contractual requirements to provide Personal Data to SAP and you fail to provide such Personal Data, then kindly note that SAP may not be able to provide you with the respective service and/or business relationship. If you or applicable law allows Us to do so, We may obtain Personal Data also from third parties which may include:</p>

      <ul>
        <li>your employer in the context of its business relationship with SAP and/or the <a href="https://www.sap.com/dam/site/corporate/legal/sap-legal-entities.pdf" target="_blank" rel="noopener noreferrer">SAP Group</a>,</li>
        <li>Third Parties you directed to share your Personal Data with SAP</li>
      </ul>

      <p>When We collect Personal Data from third parties, established internal controls aim to ensure that the third party source was permitted to provide this information to SAP and that We may use it for this purpose. SAP will treat this Personal Data according to this Privacy Statement and any additional restrictions imposed by the third party that provided the Personal Data to SAP or by applicable national law.</p>

      <h3>How long does SAP store your Personal Data?</h3>

      <p>SAP may retain your Personal Data for additional periods if necessary for compliance with legal obligations to process your Personal Data or if the Personal Data is needed by SAP to assert or defend itself against legal claims. SAP will retain your Personal Data until the end of the relevant retention period or until the claims in question have been settled. SAP does only store your Personal Data for as long as it is required:</p>

      <ul>
        <li>for SAP to comply with statutory obligations to retain Personal Data, resulting inter alia e.g. from applicable export, finance, tax or commercial laws.</li>
        <li>to process your Personal Data for this purpose and SAP obtained your consent, if required by law.</li>
      </ul>

      <h3>Who are the recipients of your Personal Data?</h3>

      <p>Your Personal Data will be transferred to or accessed by the following categories of third parties to process your Personal Data:</p>

      <ul>
        <li><strong>SAP Group entities</strong>: Other entities of the SAP Group may also receive or gain access to Personal Data either when rendering group internal services centrally and on behalf of SAP SE and the other SAP group entities or when Personal Data is transferred to them on a respective legal basis. In these cases, these entities may process the Personal Data for the same purposes and under the same conditions as outlined in this Privacy Statement. The current list of SAP Group entities can be found <a href="https://www.sap.com/dam/site/corporate/legal/sap-legal-entities.pdf" target="_blank" rel="noopener noreferrer">here</a>.</li>
      </ul>

      <h3>What are your data protection rights and how can you exercise them?</h3>

      <p>SAP honors your statutory rights when it comes to the Processing of your Personal Data. To the extent provided by applicable data protection laws, you have the right to:</p>

      <ul>
        <li>access your Personal Data that we have on you, or have it updated.</li>
        <li>Data portability of the Personal Data you provided to SAP, if SAP uses your Personal Data based on your consent or to perform a contract with you. In this case, please contact btp_ai_bp[@]sap.com and specify the information or processing activities to which your request relates, the format in which you would like to receive the Personal Data, and whether it should be sent to you or another recipient.</li>
        <li>Delete your Personal Data we hold about you. Please note, however, that SAP can or will delete your Personal Data only if there is no statutory obligation or prevailing right of SAP to retain it.</li>
        <li>Right to object against SAP further processing your Personal Data, if and to the extent SAP is processing your Personal Data based on its Legitimate Interest.</li>
        <li>Right to object to direct marketing or to apply profiling in relation to direct marketing.</li>
        <li>Revoke consent, wherever SAP is processing your Personal Data based on your consent, you may at any time withdraw your consent by unsubscribing or giving Us respective notice of withdrawal.</li>
        <li>Not to be subject to a decision based solely automated means, if the decision produces legal effects concerning you or significantly affects you in a similar way.</li>
        <li>You can request from SAP to restrict your Personal Data from further processing in any of the following events:
          <ul>
            <li>you state the Personal Data about you is incorrect, subject to the time SAP requires to check the accuracy of the relevant Personal Data,</li>
            <li>there is no legal basis for SAP to process your Personal Data and you demand SAP to restrict your Personal Data from further processing,</li>
            <li>SAP no longer requires your Personal Data, but you state you require SAP to retain such data to claim or exercise legal rights or to defend against third party claims, or</li>
            <li>in case you object to the processing of your Personal Data by SAP based on SAP's legitimate interest, subject to the time required for SAP to determine whether it has a prevailing interest or legal obligation in processing your Personal Data.</li>
          </ul>
        </li>
        <li>Lodge a complaint to the competent supervisory authority if you are not satisfied with how SAP is processing your Personal Data.</li>
      </ul>

      <h4>How you can exercise your data protection rights.</h4>

      <p>Please direct any requests to exercise your rights to btp_ai_bp[@]sap.com. SAP will take steps to ensure it verifies your identity to a reasonable degree of certainty before it will process the data protection right you want to exercise. When feasible, SAP will match Personal Data provided by you in submitting a request to exercise your rights with information already maintained by SAP.</p>

      <p>SAP will decline to process requests that are manifestly unfounded, excessive, fraudulent, represented by third parties without duly representing respective authority or are otherwise not required by local law.</p>

      <h3>Can you use SAP's services if you are a minor?</h3>

      <p>In general, the AI4U Ideation Agent is not directed to users below the age of 16 years, or equivalent minimum age in the relevant jurisdiction. If you are younger than 16 or the equivalent minimum age in the relevant jurisdiction, you cannot register with and use this website.</p>

      <h2>B. ADDITIONAL COUNTRY AND REGIONAL SPECIFIC PROVISIONS</h2>

      <h3>Where SAP is subject to privacy requirements in the EU/EEA or a country with national laws equivalent to the GDPR</h3>

      <h4>1. Who is the relevant Data Protection Authority?</h4>

      <p>You may find the contact details of your competent data protection supervisory authority <a href="https://edpb.europa.eu/about-edpb/about-edpb/members_en" target="_blank" rel="noopener noreferrer">here</a>. SAP's lead data protection supervisory authority is the Landesbeauftragter für den Datenschutz und die Informationsfreiheit Baden-Württemberg and can be reached at Lautenschlagerstraße 20, 70173 Stuttgart/Germany.</p>

      <h4>2. How does SAP justify international data transfers?</h4>

      <p>As a global group of companies, SAP has group affiliates and uses third party service providers also in countries outside the EEA. SAP may transfer your Personal Data to countries outside the EEA as part of SAP's international business operations. If We transfer Personal Data from a country in the EU or the EEA to a country outside the EEA and for which the EU Commission has not issued an adequacy decision, SAP uses the EU standard contractual clauses to contractually require the data importer to ensure a level of data protection consistent with the one in the EEA to protect your Personal Data. You may obtain a copy (redacted to remove commercial or irrelevant information) of such standard contractual clauses by sending a request to privacy[@]sap.com. You may also obtain more information from the European Commission on the international dimension of data protection <a href="https://ec.europa.eu/info/law/law-topic/data-protection/international-dimension-data-protection/standard-contractual-clauses-scc_fr" target="_blank" rel="noopener noreferrer">here</a>.</p>

      <h3>Where SAP is subject to privacy requirements in Australia</h3>

      <p>Where SAP is subject to the requirements of the Privacy Act 1988 (Cth) ('Privacy Act'), the following applies:</p>

      <p>SAP may store your Personal Data in paper-based files or as an electronic record in the Cloud or on physical devices e.g. computer systems. Your Personal Data will likely be held and stored by the <a href="https://www.sap.com/dam/site/corporate/legal/sap-legal-entities.pdf" target="_blank" rel="noopener noreferrer">SAP Group</a> located in another country for our general business purposes including outsourcing and data processing. We will only do this where it is necessary or appropriate to achieve the purposes set out in this Privacy Statement. We take reasonable steps to protect your personal information from misuse, interference and loss and from unauthorized access, modification or disclosure.</p>

      <p>You can contact Us either by the telephone number +61 2 9935 4939 or via email at btp_ai_bp[@]sap.com to exercise your rights.</p>

      <h3>Where SAP is subject to privacy requirements in the United States of America</h3>

      <p>Where SAP is subject to the requirements of the California Consumer Privacy Act of 2018 (CCPA), as amended by the California Privacy Rights Acts of 2020 (CPRA), from hereon referred to as "CCPA", or where other US State laws have similar requirements, the following applies:</p>

      <p>You have the right to:</p>

      <ul>
        <li>Know what personal information the business has collected about the consumer, including the categories of personal information, the categories of sources from which the personal information is collected, the business or commercial purpose for collecting, selling, or sharing personal information, the categories of third parties to whom the business discloses personal information, and the specific pieces of personal information the business has collected about the consumer.</li>
        <li>Delete personal information that the business has collected from the consumer, subject to certain exceptions.</li>
        <li>Correct inaccurate personal information that a business maintains about a consumer.</li>
        <li>Opt-out of the sale or sharing of their personal information by the business (where applicable).</li>
        <li>Limit the use or disclosure of sensitive personal information by the business (subject to certain exceptions, where applicable).</li>
        <li>Receive non-discriminatory treatment for the exercise of these rights.</li>
        <li>Appeal any denial of your request to exercise these rights.</li>
      </ul>

      <h4>How you can exercise your Data Protection Rights.</h4>

      <p>To exercise these rights, or to limit the Sharing of your Personal Information, please contact us at btp_ai_bp[@]sap.com or at <a href="https://support.sap.com" target="_blank" rel="noopener noreferrer">https://support.sap.com</a>.</p>

      <p><strong>Financial Incentives.</strong> SAP does not offer financial incentives in return for your consent to share your personal information, nor limit service offerings where you opt-out of such sharing (unless sharing is practically necessary to perform the relevant service).</p>

      <p><strong>Children's Privacy.</strong> Given that the AI4U Ideation Agent is not directed to users under 16 years of age, SAP does not sell or share the personal information of any minors under 16. If you are a parent or guardian and believe SAP collected information about your child, please contact SAP. SAP will take steps to delete the information as soon as possible.</p>
    </div>
  `;
}
