import type * as Preset from "@docusaurus/preset-classic";
import type { Config } from "@docusaurus/types";
import { themes as prismThemes } from "prism-react-renderer";
import { environment } from "./src/config/environment"; // Import environment config

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

// const baseUrl = environment.isDevelopment ? "/" : "/sap-btp-ai-best-practices/";
const baseUrl = "/";

function registerDownloadRoute() {
  return {
    name: "register-download-route",
    async contentLoaded({ actions }) {
      actions.addRoute({
        path: "/download-source-code/:slug*",
        component: "@site/src/pages/download-source-code.tsx",
        exact: false
      });
    }
  };
}

const config: Config = {
  title: "SAP BTP AI Best Practices",
  tagline: "Your Ultimate Guide for Building AI Solutions on SAP BTP",
  favicon: "img/favicon.ico",

  // Set the production url of your site here
  url: "https://btp-ai-bp.docs.sap/",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl,

  // Add custom fields
  customFields: {
    apiUrl: environment.apiUrl
  },

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "sap-samples", // Usually your GitHub org/user name.
  projectName: "sap-btp-ai-best-practices", // Usually your repo name.
  trailingSlash: false,

  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"]
  },

  // Add external scripts
  scripts: [
    {
      // Custom configuration for SAP's cookie consent and privacy management
      src: `${baseUrl}cookieConsentConfig.js`,
      async: false
    },
    {
      // Library designed to handle user geolocation, consent management (TrustArc), and tag management (Adobe Launch)
      src: "https://www.sap.com/sharedlibs/globaltop/script-latest.min.js",
      async: false
    }
  ],

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts"
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // editUrl:
          //   'https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ["rss", "atom"],
            xslt: true
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // editUrl:
          //   'https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/packages/create-docusaurus/templates/shared/',
          // Useful options to enforce blogging best practices
          onInlineTags: "warn",
          onInlineAuthors: "warn",
          onUntruncatedBlogPosts: "warn"
        },
        theme: {
          customCss: "./src/css/custom.css"
        }
      } satisfies Preset.Options
    ]
  ],

  plugins: [registerDownloadRoute],

  themeConfig: {
    // Social card image
    image: "img/logo.png",
    navbar: {
      title: "BTP AI Best Practices",
      logo: {
        alt: "SAP Logo",
        src: "img/logo.svg"
      },
      items: [
        { to: "/docs/functional-view", label: "Functional View", position: "left" },
        {
          // type: "docSidebar",
          // sidebarId: "tutorialSidebar",
          to: "/docs/technical-view",
          position: "left",
          label: "Technical View"
        },
        {
          to: "/about",
          position: "left",
          label: "About"
        },
        // { to: "/blog", label: "Blog", position: "left" },
        {
          type: "custom-user-dropdown",
          position: "right"
        }
        // {
        //   href: "https://github.com/SAP-samples/sap-btp-ai-best-practices",
        //   className: "navbar-icon-link",
        //   "aria-label": "GitHub repository",
        //   html: `<img src="${baseUrl}img/github.svg" alt="GitHub" width="32" height="32" />`,
        //   position: "right"
        // }
      ]
    },
    docs: {
      sidebar: {
        hideable: true
      }
    },
    footer: {
      style: "light",
      links: [
        {
          title: "SAP Logo",
          items: [
            {
              html: `
                <a href="${baseUrl}" aria-label="SAP Logo">
                  <img src="${baseUrl}img/logo.svg" alt="SAP Logo" height="50" />
                </a>`
            }
          ]
        },
        // {
        //   title: "Community",
        //   items: [
        //     {
        //       label: "GitHub",
        //       href: "https://github.com/SAP-samples/sap-btp-ai-best-practices"
        //     }
        //   ]
        // },
        {
          title: "Resources",
          items: [
            {
              label: "Technologies",
              to: `${baseUrl}docs/category/technologies`
            },
            {
              label: "Glossary",
              href: `${baseUrl}docs/glossary`
            },
            {
              label: "SAP BTP Guidance Framework",
              href: `https://discovery-center.cloud.sap/guidance-framework`
            }
          ]
        },
        {
          title: "Legal & Privacy",
          items: [
            {
              label: "Privacy Statement",
              to: "/privacy"
            },
            {
              label: "Cookie Statement",
              to: "/cookies"
            },
            {
              html: `<a class="footer__link-item open-cookie-preferences" href="#cookie-preferences">Cookie Preferences</a>`
            },
            {
              label: "Legal Disclosure",
              href: "https://www.sap.com/impressum"
            },
            {
              label: "Terms of Use",
              href: "https://www.sap.com/terms-of-use"
            },
            {
              label: "Trademark",
              href: "https://www.sap.com/trademark"
            }
          ]
        },
        {
          title: "Contact Us",
          items: [
            {
              label: "Provide Feedback",
              to: "mailto:btp_ai_bp@sap.com?subject=[SAP BTP AI Best Practices Feedback]"
            },
            {
              label: "Report Issue",
              href: "https://github.com/SAP-samples/sap-btp-ai-best-practices/issues/new"
            },
            {
              label: "Questions",
              href: "mailto:btp_ai_bp@sap.com?subject=[SAP BTP AI Best Practices Questions]"
            }
          ]
        }
      ],
      copyright: `Copyright © ${new Date().getFullYear()}, SAP SE`
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula
    }
  } satisfies Preset.ThemeConfig
};

export default config;
