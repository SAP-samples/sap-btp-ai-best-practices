/**
 * Left pane of the review page: the inbound e-mail and every document, one tab each.
 *
 * - Email tab: headers, plain-text body (HTML is never rendered) and a gallery of the e-mail's
 *   inline images; images not given to the agent are tagged with the reason.
 * - Document tabs: PDF in an <iframe>, images as <img>, Word/Excel/forwarded e-mails as text.
 * Files are fetched with the API key and shown through object URLs (revoked on unmount).
 */
import { useEffect, useState } from "react";
import { BusyIndicator, Tab, TabContainer, Tag, Text } from "@ui5/webcomponents-react";

import { fetchAttachment, type AttachmentInfo, type OrderDetail } from "../../services/orders";

/**
 * Loads an attachment as an object URL.
 *
 * @param orderId - Order id.
 * @param attachmentId - Attachment id (null = nothing to load).
 * @returns Object URL or null while loading.
 */
function useBlobUrl(orderId: string, attachmentId: string | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!attachmentId) return;
    let objectUrl: string | null = null;
    let cancelled = false;
    fetchAttachment(orderId, attachmentId)
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      })
      .catch(() => setUrl(null));
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [orderId, attachmentId]);
  return url;
}

/** Renders one stored file according to its viewer type. */
function FileView({ orderId, file }: { orderId: string; file: AttachmentInfo }) {
  const needsBlob = file.viewer === "pdf" || file.viewer === "image" || file.viewer === "download";
  const url = useBlobUrl(orderId, needsBlob ? file.attachment_id : null);
  if (file.viewer === "text") return <pre className="review-text">{file.extracted_text}</pre>;
  if (!url) return <BusyIndicator active className="review-busy" />;
  if (file.viewer === "pdf") return <iframe className="review-pdf" src={`${url}#view=FitH`} title={file.file_name} />;
  if (file.viewer === "image") return <img className="review-image" src={url} alt={file.file_name} />;
  return <a href={url} download={file.file_name}>Download {file.file_name}</a>;
}

/** Inline image with its agent-input tag. */
function InlineImage({ orderId, file }: { orderId: string; file: AttachmentInfo }) {
  const url = useBlobUrl(orderId, file.attachment_id);
  return (
    <figure className="review-inline">
      {url ? <img src={url} alt={file.file_name} /> : <BusyIndicator active size="S" />}
      <figcaption>
        <Text>{file.file_name}</Text>
        {file.model_input === "NONE" ? (
          <Tag design="Neutral" hideStateIcon title={file.model_input_reason ?? undefined}>not sent to agent</Tag>
        ) : (
          <Tag design="Information" hideStateIcon>sent to agent</Tag>
        )}
      </figcaption>
    </figure>
  );
}

/**
 * Document viewer component.
 *
 * @param props.detail - Order detail.
 * @returns Tab container with e-mail and documents.
 */
export default function DocumentViewer({ detail }: { detail: OrderDetail }) {
  const orderId = detail.header.order_id;
  const { message } = detail;
  const documents = detail.attachments.filter((a) => a.role === "DOCUMENT" || a.role === "FORWARDED_EMAIL");
  const inline = detail.attachments.filter((a) => a.role === "INLINE_IMAGE");
  const original = detail.attachments.find((a) => a.role === "SOURCE_EMAIL");
  const firstDocument = documents.find((d) => d.viewer === "pdf" || d.viewer === "image") ?? documents[0];

  return (
    <TabContainer className="review-viewer" contentBackgroundDesign="Transparent">
      <Tab text="Email" icon="email" selected={!firstDocument}>
        <div className="review-email">
          <dl>
            <dt>From</dt><dd>{message.sender ?? "-"}</dd>
            <dt>To</dt><dd>{message.recipients ?? "-"}</dd>
            <dt>Subject</dt><dd>{message.subject ?? "-"}</dd>
            <dt>Received</dt><dd>{message.received_at ?? "-"}</dd>
          </dl>
          <pre className="review-text">{message.body_text || "(no e-mail text)"}</pre>
          {inline.length > 0 && (
            <div className="review-gallery">
              {inline.map((file) => <InlineImage key={file.attachment_id} orderId={orderId} file={file} />)}
            </div>
          )}
          {original && <FileView orderId={orderId} file={original} />}
        </div>
      </Tab>
      {documents.map((file) => (
        <Tab key={file.attachment_id} text={file.file_name} selected={file === firstDocument}
          icon={file.role === "FORWARDED_EMAIL" ? "email-read" : file.viewer === "image" ? "picture" : "document"}>
          <FileView orderId={orderId} file={file} />
        </Tab>
      ))}
    </TabContainer>
  );
}
