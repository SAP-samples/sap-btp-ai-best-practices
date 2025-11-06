import React from "react";
import Layout from "@theme/Layout";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Heading from "@theme/Heading";
import PageViewTracker from "@site/src/components/tracking/PageViewTracker";
import Link from "@docusaurus/Link";
import IconLinkButton from "@site/src/components/IconLinkButton";
import Icon from "@site/src/components/Icon";
import "@ui5/webcomponents-icons/dist/idea-wall.js";
import "@ui5/webcomponents-icons/dist/developer-settings.js";
import "@ui5/webcomponents-icons/dist/tools-opportunity.js";
import "@ui5/webcomponents-icons/dist/lightbulb.js";
import "@ui5/webcomponents-icons/dist/workflow-tasks.js";

import styles from "./about.module.css";

function AboutHeader() {
  return (
    <header className={styles.heroBanner}>
      <div className={styles.videoOverlay} />
      <img src="/img/indigo-three-3d-anvils_XL.webp" alt="3D Anvils representing AI capabilities" className={styles.heroImage} />
      <div className={styles.heroContent}>
        <div className="container">
          <div className={styles.heroText}>
            <Heading as="h1" className={styles.title}>
              About
            </Heading>
            <p className={styles.subtitle}>Empowering developers to build cutting-edge, AI-powered applications on SAP BTP with proven patterns and comprehensive guidance</p>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function About(): React.ReactNode {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout title="About" description={siteConfig.tagline}>
      <AboutHeader />
      <main className={styles.main}>
        <div className="container">
          <div className={styles.contentSection}>
            <div className={styles.introSection}>
              <p className={styles.introText}>
                SAP BTP AI Best Practices is a curated collection of proven development guidelines and patterns designed to{" "}
                <strong>empower you to build cutting-edge, AI-powered applications on SAP BTP</strong>. We aim to help you accelerate your development, reduce complexity, and
                unlock new possibilities with artificial intelligence.
              </p>
            </div>

            <div className={styles.gridContainer}>
              <div className={styles.gridItem}>
                <div className={styles.card}>
                  <div className={styles.cardTitleContainer}>
                    <div className={styles.cardIcon}>
                      <Icon name="lightbulb"></Icon>
                    </div>
                    <h2>Concept</h2>
                  </div>
                  <div className={styles.cardContent}>
                    <p>
                      An AI use case leverages one or more <strong>AI capabilities</strong>, which describe what AI functionally does (e.g., summarizing text, identifying an
                      anomaly). With this in mind, we have designed the functional view of this site as a catalogue of <strong>functional patterns</strong> that frequently recur in
                      AI use cases. Each pattern comes with a clear description, practical examples, and thought-provoking questions you can use to{" "}
                      <strong>spark new use-case ideas</strong> related to that specific capability.
                    </p>
                    <p>
                      Defined AI use cases are then implemented as SAP BTP applications (whether a standalone application, an extension, or an analytics solution). We offer{" "}
                      <strong>a direct connection</strong> from the AI capabilities to implementation guidelines, leveraging one or more Best Practices to ensure your solutions are
                      robust and efficient.
                    </p>
                    <p>
                      You can also directly access the technical view, which organizes Best Practices based on different <strong>AI domains</strong> (such as generative AI or
                      narrow ML) and links them back to the possible capabilities they can implement. This view helps you understand the underlying technology and how it can be
                      applied to <strong>solve real-world business challenges</strong>.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* <div className={styles.gridContainer}>
              <div className={styles.gridItem}>
                <div className={styles.card}>
                  <div className={styles.cardTitleContainer}>
                    <div className={styles.cardIcon}>
                      <Icon name="idea-wall"></Icon>
                    </div>
                    <h2>Functional Patterns</h2>
                  </div>
                  <p>
                    An AI use case leverages one or more <strong>AI capabilities</strong>, which describe what AI functionally does (e.g., summarizing text, identifying an
                    anomaly). With this in mind, we have designed the functional view of this site as a catalogue of <strong>functional patterns</strong> that frequently recur in
                    AI use cases. Each pattern comes with a clear description, practical examples, and thought-provoking questions you can use to{" "}
                    <strong>spark new use-case ideas</strong> related to that specific capability.
                  </p>
                  <Link to="/docs/functional-view" className={styles.cardButtonLink}>
                    <span>Explore Functional Patterns</span>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </Link>
                </div>
              </div>

              <div className={styles.gridItem}>
                <div className={styles.card}>
                  <div className={styles.cardTitleContainer}>
                    <div className={styles.cardIcon}>
                      <Icon name="developer-settings"></Icon>
                    </div>
                    <h2>Technical View</h2>
                  </div>
                  <p>
                    You can also directly access the technical view, which organizes Best Practices based on different <strong>AI domains</strong> (such as generative AI or narrow
                    ML) and links them back to the possible capabilities they can implement. This view helps you understand the underlying technology and how it can be applied to{" "}
                    <strong>solve real-world business challenges</strong>.
                  </p>
                  <Link to="/docs/technical-view" className={styles.cardButtonLink}>
                    <span>Explore Technical View</span>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </Link>
                </div>
              </div>
            </div>
            <div className={styles.gridContainer}>
              <div className={styles.gridItem}>
                <div className={styles.card}>
                  <div className={styles.cardTitleContainer}>
                    <div className={styles.cardIcon}>
                      <Icon name="tools-opportunity"></Icon>
                    </div>
                    <h2>Implementation Guidelines</h2>
                  </div>
                  <p>
                    Defined AI use cases are then implemented as SAP BTP applications (whether a standalone application, an extension, or an analytics solution). We offer{" "}
                    <strong>a direct connection</strong> from the AI capabilities to implementation guidelines, leveraging one or more Best Practices to ensure your solutions are
                    robust and efficient.
                  </p>
                  <Link to="/docs/technical-view" className={styles.cardButtonLink}>
                    <span>View Implementation Guidelines</span>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </Link>
                </div>
              </div>
            </div> */}

            <div className={styles.roadmapHighlightSection}>
              <h2 className={styles.roadmapHeading}>See What's Next</h2>
              <p className={styles.roadmapDescription}>Curious about upcoming BTP AI Best Practices? Check the roadmap to see what's planned and what's coming soon!</p>
              <div className={styles.roadmapButtonWrapper}>
                <a href="/docs/SAP BTP AI Best Practices roadmap_with status_public_20251028.pdf" target="_blank" rel="noopener noreferrer" className={styles.roadmapButton}>
                  <Icon name="workflow-tasks" className={styles.roadmapButtonIcon} style={{ marginRight: 8, fontSize: 22 }} />
                  Explore the Roadmap
                </a>
              </div>
            </div>

            <div className={styles.contentSection}>
              <div className={styles.bestPracticesSection}>
                <h2>Available Best Practices</h2>
                <p>Here's a list of the SAP BTP AI Best Practices we have released so far, designed to help you get started quickly:</p>
                <div className={`button-grid ${styles.bestPracticesGrid}`}>
                  <IconLinkButton href="/docs/technical-view/generative-ai/plain/access-to-generative-ai-models" text="Access to Generative AI Models" />
                  <IconLinkButton href="/docs/technical-view/generative-ai/plain/prompt-templating" text="Prompt Templating" />
                  <IconLinkButton href="/docs/technical-view/generative-ai/plain/data-masking" text="Data Masking" />
                  <IconLinkButton href="/docs/technical-view/generative-ai/plain/content-filtering" text="Content Filtering" />
                  <IconLinkButton href="/docs/technical-view/generative-ai/rag/vector-rag-embedding" text="Vector-based RAG (1/2) Embedding" />
                  <IconLinkButton href="/docs/technical-view/generative-ai/rag/vector-rag-query-pipeline" text="Vector-based RAG (2/2) Query Pipeline" />
                  <IconLinkButton href="/docs/technical-view/generative-ai/rag/kg-rag-creation" text="Knowledge Graph RAG (1/2) Creation" />
                  <IconLinkButton href="/docs/technical-view/generative-ai/rag/kg-rag-query-pipeline" text="Knowledge Graph RAG (2/2) Query Pipeline" />
                  <IconLinkButton href="/docs/technical-view/ai-services/sap-document-ai" text="Document AI" />
                  <IconLinkButton href="/docs/technical-view/narrow-ai/regression" text="Regression" />
                  <IconLinkButton href="/docs/technical-view/narrow-ai/time-series-forecasting" text="Time Series Forecasting" />
                  <IconLinkButton href="/docs/technical-view/narrow-ai/anomaly-detection" text="Anomaly Detection" />
                  <IconLinkButton href="/docs/technical-view/ai-services/custom-joule-skills" text="Custom Joule Skills" />
                </div>
                <p>
                  Further SAP BTP AI Best Practices are in preparation. <strong>Stay tuned</strong> to keep your AI development on SAP BTP at the forefront!
                </p>
              </div>
            </div>
          </div>
        </div>
        <div className={styles.ctaSectionWrapper}>
          <div className={styles.ctaSection}>
            <div className={styles.ctaCard}>
              <h2>Get Involved</h2>
              <p>
                We're continuously expanding this collection to bring you the latest insights and proven approaches. Your feedback is invaluable in shaping this resource, so please
                give us your feedback.
              </p>
              <Link to="mailto:btp_ai_bp@sap.com?subject=[SAP BTP AI Best Practices Feedback]" className={`${styles.cardButtonLink} ${styles.cardButtonLinkWhite}`}>
                <span>Provide Feedback</span>
              </Link>
            </div>
          </div>
        </div>
      </main>
      <PageViewTracker />
    </Layout>
  );
}
