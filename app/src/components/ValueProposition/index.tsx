import Heading from "@theme/Heading";
import clsx from "clsx";
import type { ReactNode } from "react";
import styles from "./styles.module.css";

type ValueItem = {
  title: string;
  description: ReactNode;
  icon: ReactNode;
};

const ValueList: ValueItem[] = [
  {
    title: "Comprehensive Guidance",
    description: "Step-by-step guidance for implementing AI solutions on SAP Business AI Platform, covering both functional and technical aspects.",
    icon: (
      <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M24 4L4 14V34L24 44L44 34V14L24 4Z" fill="currentColor" fillOpacity="0.1" />
        <path d="M24 4L4 14V34L24 44L44 34V14L24 4Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M24 4V44" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M4 14L44 14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M4 34L44 34" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    )
  },
  {
    title: "Proven Patterns",
    description: "Battle-tested patterns and architectures for scalable AI solutions on SAP Business AI Platform.",
    icon: (
      <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M8 8H40V40H8V8Z" fill="currentColor" fillOpacity="0.1" />
        <path d="M8 8H40V40H8V8Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M16 16H32V32H16V16Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M24 8V40" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M8 24H40" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    )
  },
  {
    title: "Accelerated Development",
    description: "Reduce development time with comprehensive documentation and proven methodologies.",
    icon: (
      <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M24 4L4 14V34L24 44L44 34V14L24 4Z" fill="currentColor" fillOpacity="0.1" />
        <path d="M24 4L4 14V34L24 44L44 34V14L24 4Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M24 4V44" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M4 14L44 14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M4 34L44 34" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    )
  },
  {
    title: "Comprehensive Guidance",
    description: "Step-by-step guidance for implementing AI solutions on SAP Business AI Platform, covering both functional and technical aspects.",
    icon: (
      <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M24 4L4 14V34L24 44L44 34V14L24 4Z" fill="currentColor" fillOpacity="0.1" />
        <path d="M24 4L4 14V34L24 44L44 34V14L24 4Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M24 4V44" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M4 14L44 14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M4 34L44 34" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    )
  },
  {
    title: "Proven Patterns",
    description: "Battle-tested patterns and architectures for scalable AI solutions on SAP Business AI Platform.",
    icon: (
      <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M8 8H40V40H8V8Z" fill="currentColor" fillOpacity="0.1" />
        <path d="M8 8H40V40H8V8Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M16 16H32V32H16V16Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M24 8V40" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M8 24H40" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    )
  },
  {
    title: "Accelerated Development",
    description: "Reduce development time with comprehensive documentation and proven methodologies.",
    icon: (
      <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M24 4L4 14V34L24 44L44 34V14L24 4Z" fill="currentColor" fillOpacity="0.1" />
        <path d="M24 4L4 14V34L24 44L44 34V14L24 4Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M24 4V44" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M4 14L44 14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M4 34L44 34" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    )
  }
];

function ValueItem({ title, description, icon }: ValueItem) {
  return (
    <div className={clsx(styles.valueItem)}>
      <div className={styles.icon}>{icon}</div>
      <Heading as="h3" className={styles.title}>
        {title}
      </Heading>
      <p className={styles.description}>{description}</p>
    </div>
  );
}

export default function ValueProposition(): ReactNode {
  return (
    <section className={styles.valueProposition}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <Heading as="h2" className="section-title">
              Why Use SAP Business AI Platform <span className="highlight">Best Practices</span>?
            </Heading>
          </div>
        </div>
        <div
          style={{
            position: "relative",
            zIndex: 1,
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "space-between",
            margin: "70px -1rem",
            marginBottom: "100px"
          }}
        >
          {ValueList.map((props, idx) => (
            <ValueItem key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
