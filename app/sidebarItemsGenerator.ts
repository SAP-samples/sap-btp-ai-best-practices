import type { Options } from "@docusaurus/plugin-content-docs";

const sidebarItemsGenerator: NonNullable<Options["sidebarItemsGenerator"]> = async ({ defaultSidebarItemsGenerator, ...args }) => {
  const generatedItems = await defaultSidebarItemsGenerator(args);
  type Items = typeof generatedItems;
  const agenticPageOrder = [
    "technical-view/agentic-ai/concepts/agentic-ai-concepts-explained",
    "technical-view/agentic-ai/concepts/joule-studio-ai-agents",
    "technical-view/agentic-ai/concepts/pro-code-ai-agents",
    "technical-view/agentic-ai/concepts/joule-agent-builder"
  ];

  function flattenAgenticAI(items: Items): Items {
    return items.flatMap((item): Items => {
      if (item.type !== "category") return [item];
      if (item.label !== "Agentic AI Concepts") return [{ ...item, items: flattenAgenticAI(item.items) }];

      return [...item.items].sort((left, right) => {
        const position = (entry: Items[number]) => {
          const index = entry.type === "doc" ? agenticPageOrder.indexOf(entry.id) : -1;
          return index < 0 ? agenticPageOrder.length : index;
        };
        return position(left) - position(right);
      });
    });
  }

  return flattenAgenticAI(generatedItems);
};

export default sidebarItemsGenerator;
