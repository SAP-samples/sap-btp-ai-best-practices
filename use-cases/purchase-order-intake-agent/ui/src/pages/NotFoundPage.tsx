/**
 * Fallback page for unknown URLs (route "*").
 */
import { useNavigate } from "react-router";
import { Button, Text, Title } from "@ui5/webcomponents-react";

import { DEFAULT_PATH } from "../routes";

/**
 * Renders a "Page Not Found" message with a button back to the default page.
 *
 * @returns The not-found view.
 */
export default function NotFoundPage() {
  const navigate = useNavigate();

  return (
    <div style={{ padding: "2rem", textAlign: "center" }}>
      <Title level="H2">Page Not Found</Title>
      <Text>The requested page could not be found.</Text>
      <div style={{ marginTop: "1rem" }}>
        <Button onClick={() => navigate(DEFAULT_PATH)}>Go to Purchase Orders</Button>
      </div>
    </div>
  );
}
