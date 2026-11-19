import React, { type ReactNode } from "react";
import clsx from "clsx";
import TrackableLink from "@site/src/components/TrackableLink";
import styles from "./styles.module.css";

const QUALTRICS_FORM = "https://sapxm.qualtrics.com/jfe/form/SV_2ozSvl0Byxi56Ki";

const buildUrl = (utmContent: string): string =>
  `${QUALTRICS_FORM}?utm_source=btp-ai-best-practices&utm_medium=referral&utm_campaign=baip-activation&utm_content=${utmContent}`;

interface ActivationCTAProps {
  /** Distinguishes the placement via the Qualtrics `utm_content` param, e.g. "homepage-band". */
  utmContent: string;
  /** Tighter padding/typography for dense placements such as the docs page footer. */
  compact?: boolean;
}

/**
 * Full-width call-to-action band inviting customers to activate SAP Business AI
 * Platform services in their own BTP tenant via the Qualtrics onboarding form.
 * Links out in a new tab (the form carries its own privacy/consent gate).
 */
export default function ActivationCTA({ utmContent, compact = false }: ActivationCTAProps): ReactNode {
  return (
    <section className={clsx(styles.wrapper, compact && styles.wrapperCompact)}>
      <div className={clsx(styles.card, compact && styles.cardCompact)}>
        <div className={styles.content}>
          <p className={styles.eyebrow}>SAP Business AI Platform</p>
          <h2 className={styles.title}>Interested in activating AI in your Business AI Platform environment?</h2>
          <p className={styles.subtitle}>
            Tell us about your needs and the SAP Business AI Platform onboarding team will help you switch on AI services in your BTP tenant.
          </p>
        </div>
        <TrackableLink
          href={buildUrl(utmContent)}
          className={styles.button}
          target="_blank"
          rel="noopener noreferrer"
          trackingFeature="ACTIVATION_QUALTRICS"
        >
          <span>Get started</span>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </TrackableLink>
      </div>
    </section>
  );
}
