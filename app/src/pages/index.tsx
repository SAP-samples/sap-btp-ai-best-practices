import Link from "@docusaurus/Link";
import useBaseUrl from "@docusaurus/useBaseUrl";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Heading from "@theme/Heading";
import Layout from "@theme/Layout";
import clsx from "clsx";
import { type ReactNode } from "react";

import HomepageSections from "../components/HomepageSections";
// import ValueProposition from "../components/ValueProposition";
import ActivationCTA from "../components/ActivationCTA";
import PageViewTracker from "../components/tracking/PageViewTracker";

import styles from "./index.module.css";

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();

  return (
    <header className={clsx("hero hero--primary", styles.heroBanner)}>
      <div className={styles.videoOverlay} />
      <video className={styles.videoBackground} autoPlay loop muted playsInline>
        <source src={useBaseUrl("/videos/business-ai-video-hero.mp4")} type="video/mp4" />
      </video>
      <div className={clsx("container", styles.heroContent)}>
        <div className={styles.heroText}>
          <Heading as="h1" className={styles.title}>
            SAP BTP <span className={styles.highlight}>AI Best Practices</span>
          </Heading>
          <p className={styles.subtitle}>{siteConfig.tagline}</p>
        </div>
        <p className={styles.description}>
          SAP BTP AI Best Practices is a curated collection of guides and patterns for implementing AI solutions on SAP Business Technology Platform. Explore a{" "}
          <b>functional view</b> that showcases what AI can do, and a <b>technical view</b> that shows how to make it real.
        </p>
        <div className={styles.navigation}>
          {/* <p className={styles.navTitle}>Choose your path:</p> */}
          <div className={styles.navLinks}>
            <Link to="/docs/functional-view" className={styles.navLink}>
              <span>Functional View</span>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </Link>
            <Link to="/docs/technical-view" className={styles.navLink}>
              <span>Technical View</span>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout description={siteConfig.tagline}>
      <HomepageHeader />
      <main className={styles.main}>
        <HomepageSections />
        {/* <ValueProposition /> */}
        <ActivationCTA utmContent="homepage-band" />
      </main>

      <PageViewTracker />
    </Layout>
  );
}
