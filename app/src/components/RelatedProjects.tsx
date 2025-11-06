import React from "react";
import { useLocation } from "@docusaurus/router";
import IconLinkButton from "./IconLinkButton";
import { projects } from "../data/projects.js";

interface RelatedProjectsProps {
  className?: string;
}

const RelatedProjects: React.FC<RelatedProjectsProps> = ({ className = "button-grid" }) => {
  const location = useLocation();

  // Get current page path and construct the expected URL
  const currentPath = location.pathname;

  // Convert Docusaurus path to the expected URL format in relatedBestPractices
  // e.g., /docs/technical-view/generative-ai/plain/access-to-generative-ai-models
  // becomes https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models
  const expectedUrl = `https://btp-ai-bp.docs.sap${currentPath}`;

  // Filter projects that have the current page URL in their relatedBestPractices
  const relatedProjects = projects.filter((project) => {
    if (!project.relatedBestPractices || !Array.isArray(project.relatedBestPractices)) {
      return false;
    }

    return project.relatedBestPractices.some((bestPractice) => bestPractice.url === expectedUrl);
  });

  // If no related projects found, return null to hide the section
  if (relatedProjects.length === 0) {
    return null;
  }

  return (
    <>
      <br />

      <div className="section-with-background blue">
        <h2>
          <span className="post-article-first-title">Example Projects & Use Cases</span>
        </h2>

        <ul className={className}>
          {relatedProjects.map((project) => (
            <IconLinkButton key={project.id} href={`https://ai4u-website.cfapps.eu10-004.hana.ondemand.com/project/${project.id}`} text={project.title} icon={project.icon} />
          ))}
        </ul>
      </div>
    </>
  );
};

export default RelatedProjects;
