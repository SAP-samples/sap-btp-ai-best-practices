import React, { type ReactNode } from "react";
import Footer from "@theme-original/DocItem/Footer";
import { useLocation } from "@docusaurus/router";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import ActivationCTA from "@site/src/components/ActivationCTA";

/**
 * Derives a stable, readable `utm_content` for the current doc page, e.g.
 * "docs-footer:functional-view/ai-services/sap-rpt". The placement prefix keeps
 * it groupable with the other activation surfaces; the page slug reveals which
 * best-practice content drives activations.
 */
function useDocsFooterUtmContent(): string {
  const { pathname } = useLocation();
  const { siteConfig } = useDocusaurusContext();

  let slug = pathname;
  if (slug.startsWith(siteConfig.baseUrl)) {
    slug = slug.slice(siteConfig.baseUrl.length);
  }
  slug = slug.replace(/^docs\//, "").replace(/\/$/, "");

  return slug ? `docs-footer:${slug}` : "docs-footer";
}

/**
 * Wraps the default docs page footer and appends the activation CTA band, so
 * customers are invited to switch on SAP Business AI Platform services right
 * after reading about them. Rendered on every doc page.
 */
export default function FooterWrapper(props: Record<string, unknown>): ReactNode {
  const utmContent = useDocsFooterUtmContent();

  return (
    <>
      <Footer {...props} />
      <ActivationCTA utmContent={utmContent} compact />
    </>
  );
}
