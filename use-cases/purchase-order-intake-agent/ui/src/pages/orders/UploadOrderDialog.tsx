/**
 * "Upload Purchase Order" dialog: drop zone / file browser, optional pasted e-mail text,
 * optional sales organisation (simulates the inbox-to-sales-org mapping), Extract button.
 *
 * Every .eml file becomes its own order; other files are processed together as one order
 * (the first file is the purchase order, the others supporting files).
 */
import { useEffect, useState, type DragEvent } from "react";
import {
  Bar,
  Button,
  Dialog,
  FileUploader,
  Label,
  Link,
  MessageStrip,
  Option,
  Select,
  Text,
  TextArea
} from "@ui5/webcomponents-react";

import { listSalesOrgs, uploadOrders } from "../../services/orders";

const ACCEPT = ".eml,.pdf,.png,.jpg,.jpeg,.gif,.webp,.docx,.xlsx,.txt,.csv";

interface Props {
  /** Whether the dialog is shown. */
  open: boolean;
  /** Called when the dialog closes (cancel or after a successful upload). */
  onClose: () => void;
  /** Called with the number of created orders after a successful upload. */
  onUploaded: (count: number) => void;
}

/**
 * Upload dialog component.
 *
 * @param props - See `Props`.
 * @returns The dialog element.
 */
export default function UploadOrderDialog({ open, onClose, onUploaded }: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [emailText, setEmailText] = useState("");
  const [salesOrg, setSalesOrg] = useState("");
  const [salesOrgs, setSalesOrgs] = useState<string[]>([]);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) listSalesOrgs().then(setSalesOrgs).catch(() => setSalesOrgs([]));
  }, [open]);

  const reset = () => {
    setFiles([]);
    setEmailText("");
    setError(null);
    setBusy(false);
  };

  const addFiles = (list: FileList | null) => {
    if (list) setFiles((current) => [...current, ...Array.from(list)]);
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    addFiles(event.dataTransfer.files);
  };

  const extract = async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await uploadOrders(files, emailText, salesOrg);
      onUploaded(result.orders.length);
      reset();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  };

  return (
    <Dialog
      open={open}
      headerText="Upload Purchase Order"
      className="upload-dialog"
      onClose={() => {
        reset();
        onClose();
      }}
      footer={
        <Bar
          design="Footer"
          endContent={
            <>
              <Button design="Emphasized" disabled={busy || (!files.length && !emailText.trim())} onClick={extract}>
                {busy ? "Uploading..." : "Extract"}
              </Button>
              <Button design="Transparent" onClick={() => { reset(); onClose(); }}>
                Cancel
              </Button>
            </>
          }
        />
      }
    >
      <div
        className={`upload-dropzone${dragging ? " upload-dropzone--active" : ""}`}
        onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <Text>Drop files here, or</Text>
        <FileUploader hideInput multiple accept={ACCEPT} onChange={(event) => addFiles(event.detail.files)}>
          <Link>Browse</Link>
        </FileUploader>
        <Text className="upload-hint">
          PDF, image, Word, Excel or .eml. Each e-mail becomes its own order; other files are processed together
          (the first file is the purchase order, the others are supporting files).
        </Text>
        {files.length > 0 && (
          <ul className="upload-files">
            {files.map((file, index) => (
              <li key={`${file.name}-${index}`}>
                {file.name} ({Math.ceil(file.size / 1024)} KB)
                <Button design="Transparent" icon="decline" tooltip="Remove"
                  onClick={() => setFiles((current) => current.filter((_, i) => i !== index))} />
              </li>
            ))}
          </ul>
        )}
      </div>
      <Label for="email-text">Email Content (optional):</Label>
      <TextArea id="email-text" rows={5} placeholder="Paste the customer's email text here" value={emailText}
        onInput={(event) => setEmailText(event.target.value)} />
      <Label for="sales-org">Sales Organisation (inbox mapping, optional):</Label>
      <Select id="sales-org" onChange={(event) => setSalesOrg(event.detail.selectedOption.dataset.value ?? "")}>
        <Option data-value="" selected={salesOrg === ""}>Determine automatically</Option>
        {salesOrgs.map((org) => (
          <Option key={org} data-value={org} selected={salesOrg === org}>{org}</Option>
        ))}
      </Select>
      {error && <MessageStrip design="Negative" hideCloseButton className="upload-error">{error}</MessageStrip>}
    </Dialog>
  );
}
