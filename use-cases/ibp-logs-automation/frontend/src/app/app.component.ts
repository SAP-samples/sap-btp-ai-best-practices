import {
  Component, OnInit, OnDestroy, signal, computed,
  ViewChild, TemplateRef, inject
} from '@angular/core';
import { CommonModule, DatePipe } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { interval, Subscription } from 'rxjs';
import { switchMap } from 'rxjs/operators';

import { ShellbarModule }      from '@fundamental-ngx/core/shellbar';
import { ButtonModule }        from '@fundamental-ngx/core/button';
import { BusyIndicatorModule } from '@fundamental-ngx/core/busy-indicator';
import { MessageStripModule }  from '@fundamental-ngx/core/message-strip';
import { TableModule }         from '@fundamental-ngx/core/table';
import { DialogModule, DialogService } from '@fundamental-ngx/core/dialog';
import { IconModule }          from '@fundamental-ngx/core/icon';
import { ScrollbarModule }     from '@fundamental-ngx/core/scrollbar';
import { FormsModule }         from '@angular/forms';

// CF backend — file storage, HANA, session management (always CF)
// Replace <your-cf-backend-host> with your deployed backend URL, or wire this
// value through Angular's environment.ts files for per-environment builds.
const API       = 'https://<your-cf-backend-host>/api';
// Local agent — runs extract.py with Chrome (must be running on the user's machine)
const LOCAL_API = 'http://localhost:5001/api';

interface CsvFileInfo {
  filename: string;
  csvName: string;
  rowCount: number;
  columns: string[];
  sizeBytes: number;
  checksum: string;
  extractedAt: string;
}

interface LogFile {
  filename: string;
  sizeBytes: number;
  modifiedAt: string;
  checksum: string;
  jobName: string;
  csvFiles: CsvFileInfo[];
}

interface JobStatus {
  status: 'idle' | 'running' | 'done' | 'error';
  message: string;
  file: string | null;
  csvFiles: CsvFileInfo[];
  log: { ts: string; msg: string }[];
  hanaStatus: Record<string, unknown> | null;
  jobName: string;
}

interface CsvData {
  filename: string;
  columns: string[];
  rows: Record<string, string>[];
  totalRows: number;
}

interface ModeInfo { canExtract: boolean; platform: string; }
interface SessionInfo { hasCookies: boolean; cookieCount: number; savedAt: string | null; }

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule, DatePipe, FormsModule,
    ShellbarModule, ButtonModule, BusyIndicatorModule,
    MessageStripModule, TableModule, DialogModule,
    IconModule, ScrollbarModule,
  ],
  providers: [DialogService, DatePipe],
  template: `
    <!-- ── Shell Bar ──────────────────────────────────────────────── -->
    <fd-shellbar>
      <fd-shellbar-logo>
        <a class="fd-shellbar__logo fd-shellbar__logo--image-replaced" aria-label="SAP">
          <img src="sap-logo.png" alt="SAP" style="height:30px;width:auto;display:block">
        </a>
      </fd-shellbar-logo>
      <fd-shellbar-title>IBP Optimizer Log Extractor</fd-shellbar-title>
      <fd-shellbar-actions>
        <fd-shellbar-action>
          <span style="color:white;font-size:13px;opacity:.8;padding-right:16px">
            SAP IBP &nbsp;·&nbsp; Tenant IBP1
          </span>
        </fd-shellbar-action>
      </fd-shellbar-actions>
    </fd-shellbar>

    <div style="padding:1.5rem 2.5rem;">

      <!-- Page header -->
      <div style="margin-bottom:1.25rem;">
        <h1 style="font-size:1.25rem;font-weight:700;color:var(--sapTextColor,#32363a);margin:0 0 .25rem">
          Optimizer Log Extraction
        </h1>
        <p style="font-size:.875rem;color:var(--sapNeutralTextColor,#6a6d70);margin:0">
          Automates retrieval of Supply Planning Logs from the latest finished optimizer run.
          CSV data is stored in SAP HANA and available for preview below.
        </p>
      </div>

      <!-- ── Run control card ──────────────────────────────────────── -->
      <div class="ibp-card" style="margin-bottom:1.25rem;padding:1.25rem 1.5rem;">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
          <div style="flex:1;min-width:280px;">
            <label class="fd-form-label" for="jobNameInput"
              style="font-size:.8125rem;font-weight:600;display:block;margin-bottom:.375rem">
              Job Name
            </label>
            <input class="fd-input" id="jobNameInput"
              [value]="jobNameInput()"
              (input)="jobNameInput.set($any($event.target).value)"
              placeholder="Enter Application Job name..."
              [disabled]="isRunning()"
              style="width:100%;max-width:520px">
            <div style="font-size:.75rem;color:var(--sapNeutralTextColor,#6a6d70);margin-top:.25rem">
              Attachment: <code style="font-size:.75rem">Supply Planning Logs</code>
              &nbsp;·&nbsp; Target: <code style="font-size:.75rem">/IBP/OptimizerLogs/IBP1/TS/</code>
            </div>

            <!-- Session status -->
            @if (sessionInfo()) {
              <div style="display:flex;align-items:center;gap:.5rem;margin-top:.625rem;flex-wrap:wrap;">
                @if (sessionInfo()!.hasCookies) {
                  <fd-icon glyph="sys-enter-2" style="color:var(--sapPositiveColor,#188918);font-size:.875rem"></fd-icon>
                  <span style="font-size:.8125rem;color:var(--sapPositiveColor,#188918);font-weight:600">
                    Session active — {{ sessionInfo()!.cookieCount }} cookies
                  </span>
                  @if (sessionInfo()!.savedAt) {
                    <span style="font-size:.75rem;color:var(--sapNeutralTextColor,#6a6d70)">
                      · uploaded {{ sessionInfo()!.savedAt | date:'dd MMM HH:mm' }}
                    </span>
                  }
                } @else {
                  <fd-icon glyph="warning" style="color:var(--sapCriticalColor,#e9730c);font-size:.875rem"></fd-icon>
                  <span style="font-size:.8125rem;color:var(--sapCriticalColor,#e9730c);font-weight:600">No session cookies</span>
                  <span style="font-size:.75rem;color:var(--sapNeutralTextColor,#6a6d70)">Upload cookies to authenticate</span>
                }

                <!-- Upload cookies button — always visible -->
                <label style="cursor:pointer;margin-left:.25rem">
                  <button fd-button fdType="emphasized" glyph="key"
                    [label]="uploadingCookies() ? 'Uploading…' : 'Upload IBP Cookies'"
                    [disabled]="uploadingCookies()"
                    (click)="cookieInput.click()"
                    ariaLabel="Upload IBP session cookies JSON">
                  </button>
                  <input #cookieInput type="file" accept=".json"
                    style="display:none"
                    (change)="uploadCookies($event)">
                </label>

                @if (cookieUploadResult()) {
                  <div style="font-size:.75rem;margin-top:.25rem"
                    [style.color]="cookieUploadResult()!.error ? 'var(--sapNegativeColor,#bb0000)' : 'var(--sapPositiveColor,#188918)'">
                    {{ cookieUploadResult()!.error ?? '✓ ' + cookieUploadResult()!.count + ' cookies uploaded — ready to extract' }}
                  </div>
                }

                <div style="font-size:.7rem;color:var(--sapNeutralTextColor,#8a8a8a);margin-top:.25rem">
                  Login to IBP in Chrome → install <strong>Cookie-Editor</strong> extension → Export as JSON → upload here
                </div>
              </div>
            }
          </div>

          <div style="display:flex;align-items:center;gap:.75rem;flex-shrink:0">
            <span class="fd-object-status"
              [class.fd-object-status--positive]="jobStatus()?.status === 'done'"
              [class.fd-object-status--negative]="jobStatus()?.status === 'error'"
              [class.fd-object-status--informative]="jobStatus()?.status === 'running'"
              [class.fd-object-status--neutral]="!jobStatus() || jobStatus()?.status === 'idle'">
              <fd-icon [glyph]="statusIcon()" class="fd-object-status__icon"></fd-icon>
              <span class="fd-object-status__text">{{ statusLabel() }}</span>
            </span>

            @if (canExtract()) {
              <!-- Local agent running: trigger extraction -->
              <fd-busy-indicator [loading]="isRunning()" ariaLabel="Extracting" size="s">
                <button fd-button fdType="emphasized"
                  [label]="isRunning() ? 'Extracting…' : 'Extract Now'"
                  glyph="download" [disabled]="isRunning()"
                  (click)="triggerRun()" ariaLabel="Start extraction">
                </button>
              </fd-busy-indicator>
            } @else {
              <!-- Local agent not running + Kyma not configured -->
              <div style="display:flex;align-items:center;gap:.625rem;flex-wrap:wrap;">
                <fd-icon glyph="laptop" style="font-size:1.1rem;color:var(--sapCriticalColor,#e9730c)"></fd-icon>
                <div>
                  <div style="font-size:.875rem;font-weight:600;color:var(--sapCriticalColor,#e9730c)">
                    Local agent not running
                  </div>
                  <div style="font-size:.75rem;color:var(--sapNeutralTextColor,#6a6d70)">
                    Open <strong>IBP-Agent.app</strong> on your Mac or run <strong>IBP-Agent.bat</strong> on Windows
                  </div>
                </div>
              </div>
            }
          </div>
        </div>

        @if (lastMessage()) {
          <div style="margin-top:1rem;">
            <fd-message-strip [type]="messageType()" [dismissible]="false">
              {{ lastMessage() }}
            </fd-message-strip>
          </div>
        }

        <!-- HANA status badge + Clear button -->
        @if (jobStatus()?.hanaStatus && jobStatus()?.status === 'done') {
          <div style="margin-top:.75rem;display:flex;align-items:center;gap:.5rem;font-size:.8125rem;flex-wrap:wrap;">
            <fd-icon glyph="database" style="color:var(--sapPositiveColor,#188918)"></fd-icon>
            <span style="color:var(--sapPositiveColor,#188918);font-weight:600">
              Data loaded into SAP HANA
            </span>
            @for (entry of hanaEntries(); track entry.file) {
              <span style="color:var(--sapNeutralTextColor,#6a6d70)">
                &nbsp;·&nbsp; {{ entry.file }} → <code style="font-size:.75rem">{{ entry.table }}</code>: {{ entry.inserted }} rows
              </span>
            }
          </div>
        }
      </div>

      <!-- ── Progress panel ────────────────────────────────────────── -->
      @if ((isRunning() || jobStatus()?.status === 'error') && runLog().length > 0) {
        <div class="ibp-card" style="margin-bottom:1.25rem;padding:1rem 1.25rem;">
          <div class="ibp-section-label">Extraction Progress</div>
          <div fd-scrollbar style="max-height:20rem;overflow-y:auto;" #progressPanel>
            @for (entry of runLog(); track entry.ts) {
              <div style="display:flex;gap:.625rem;font-family:monospace;font-size:.75rem;padding:.125rem 0;">
                <span style="color:var(--sapNeutralTextColor,#8a8a8a);white-space:nowrap">
                  {{ formatTime(entry.ts) }}
                </span>
                <span style="color:var(--sapTextColor,#32363a)">{{ entry.msg }}</span>
              </div>
            }
          </div>
        </div>
      }

      <!-- ── Latest extraction CSVs (shown right after done) ──────── -->
      @if (jobStatus()?.status === 'done' && jobStatus()?.csvFiles?.length) {
        <div style="margin-bottom:1.25rem;">
          <div style="font-size:.9375rem;font-weight:600;color:var(--sapTextColor,#32363a);margin-bottom:.75rem">
            Extracted Files
          </div>
          <div style="display:flex;gap:1rem;flex-wrap:wrap;">
            @for (csv of jobStatus()!.csvFiles; track csv.csvName) {
              <div class="ibp-card ibp-csv-chip" (click)="openCsvViewer(jobStatus()!.file!, csv.csvName)">
                <fd-icon glyph="document-text" style="font-size:1.25rem;color:var(--sapLinkColor,#0a6ed1)"></fd-icon>
                <div>
                  <div style="font-weight:600;font-size:.875rem">{{ csv.csvName }}</div>
                  <div style="font-size:.75rem;color:var(--sapNeutralTextColor,#6a6d70)">
                    {{ csv.rowCount | number }} rows · {{ formatBytes(csv.sizeBytes) }}
                  </div>
                </div>
              </div>
            }
          </div>
        </div>
      }

      <!-- ── Downloaded files table ────────────────────────────────── -->
      <div class="ibp-card" style="overflow:hidden;margin-bottom:1.25rem;">
        <div style="display:flex;align-items:center;justify-content:space-between;padding:.875rem 1.25rem;border-bottom:1px solid var(--sapGroup_TitleBorderColor,#e5e5e5);">
          <span style="font-size:.9375rem;font-weight:600;color:var(--sapTextColor,#32363a)">
            Downloaded Log Files
            @if (files().length > 0) {
              <span class="ibp-badge">{{ files().length }}</span>
            }
          </span>
          <button fd-button fdType="transparent" glyph="refresh" label="Refresh"
            (click)="loadFiles()" ariaLabel="Refresh"></button>
        </div>

        @if (files().length === 0) {
          <div style="padding:3.5rem 2rem;text-align:center;color:var(--sapNeutralTextColor,#8a8a8a);">
            <fd-icon glyph="documents" style="font-size:3rem;display:block;margin-bottom:.875rem;"></fd-icon>
            <div style="font-size:.9375rem;font-weight:600;margin-bottom:.375rem">No log files available</div>
            <div style="font-size:.875rem">Click <strong>Extract Now</strong> to retrieve the latest optimizer log.</div>
          </div>
        } @else {
          <table fd-table aria-label="Downloaded optimizer log files">
            <thead fd-table-header>
              <tr fd-table-row>
                <th fd-table-cell>File</th>
                <th fd-table-cell>Downloaded</th>
                <th fd-table-cell>Size</th>
                <th fd-table-cell>CSV Contents</th>
                <th fd-table-cell>Actions</th>
              </tr>
            </thead>
            <tbody fd-table-body>
              @for (file of files(); track file.filename) {
                <tr fd-table-row>
                  <td fd-table-cell>
                    <fd-icon glyph="attachment-zip" style="margin-right:.375rem;color:var(--sapLinkColor,#0a6ed1)"></fd-icon>
                    {{ file.filename }}
                  </td>
                  <td fd-table-cell>{{ file.modifiedAt | date:'dd MMM yyyy, HH:mm' }}</td>
                  <td fd-table-cell>{{ formatBytes(file.sizeBytes) }}</td>
                  <td fd-table-cell>
                    <div style="display:flex;gap:.5rem;flex-wrap:wrap;">
                      @for (csv of file.csvFiles; track csv.csvName) {
                        <button fd-button fdType="ghost" [label]="csv.csvName"
                          (click)="loadCsvInline(file.filename, csv.csvName)"
                          [class.ibp-active-btn]="activeCsvKey() === file.filename + csv.csvName"
                          style="font-size:.75rem">
                        </button>
                      }
                    </div>
                  </td>
                  <td fd-table-cell>
                    <a [href]="API + '/files/' + file.filename + '/download'" target="_blank">
                      <button fd-button fdType="transparent" glyph="download"
                        ariaLabel="Download" title="Download"></button>
                    </a>
                  </td>
                </tr>
              }
            </tbody>
          </table>
        }
      </div>

      <!-- ── Inline CSV data table ──────────────────────────────────── -->
      @if (selectedCsv()) {
        <div class="ibp-card" style="overflow:hidden;margin-bottom:1.25rem;">

          <!-- Header -->
          <div style="display:flex;align-items:center;justify-content:space-between;
                      padding:.875rem 1.25rem;border-bottom:1px solid var(--sapGroup_TitleBorderColor,#e5e5e5);">
            <div>
              <span style="font-size:.9375rem;font-weight:600;color:var(--sapTextColor,#32363a)">
                {{ selectedCsv()!.filename }}
              </span>
              <span style="font-size:.8125rem;color:var(--sapNeutralTextColor,#6a6d70);margin-left:.75rem">
                {{ selectedCsv()!.totalRows | number }} rows &nbsp;·&nbsp;
                {{ selectedCsv()!.columns.length }} columns
                @if (selectedCsv()!.totalRows > 500) {
                  &nbsp;·&nbsp; <em>showing first 500</em>
                }
              </span>
            </div>
            <button fd-button fdType="transparent" glyph="decline"
              (click)="selectedCsv.set(null)" ariaLabel="Close" title="Close"></button>
          </div>

          <!-- Scrollable table -->
          <div style="overflow:auto;max-height:480px;">
            <table fd-table aria-label="CSV data" style="font-size:.75rem;white-space:nowrap;min-width:100%">
              <thead fd-table-header>
                <tr fd-table-row>
                  @for (col of selectedCsv()!.columns; track col) {
                    <th fd-table-cell style="font-size:.75rem;font-weight:600;
                        position:sticky;top:0;background:var(--sapList_HeaderBackground,#f2f2f2);
                        z-index:1">
                      {{ col }}
                    </th>
                  }
                </tr>
              </thead>
              <tbody fd-table-body>
                @for (row of selectedCsv()!.rows; track $index) {
                  <tr fd-table-row [hoverable]="true">
                    @for (col of selectedCsv()!.columns; track col) {
                      <td fd-table-cell style="font-size:.75rem;
                          max-width:260px;overflow:hidden;text-overflow:ellipsis"
                          [title]="row[col]">
                        {{ row[col] }}
                      </td>
                    }
                  </tr>
                }
              </tbody>
            </table>
          </div>

        </div>
      }

      <!-- ── HANA management card ───────────────────────────────────── -->
      <div class="ibp-card" style="padding:1rem 1.5rem;margin-bottom:1.25rem;
           display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
        <div style="display:flex;align-items:center;gap:.625rem;">
          <fd-icon glyph="database" style="font-size:1.1rem;color:var(--sapNeutralTextColor,#6a6d70)"></fd-icon>
          <div>
            <div style="font-size:.875rem;font-weight:600;color:var(--sapTextColor,#32363a)">
              SAP HANA — {{ HANA_TABLE }}
            </div>
            <div style="font-size:.75rem;color:var(--sapNeutralTextColor,#6a6d70)">
              Schema: {{ HANA_SCHEMA }}
              @if (clearResult()) {
                &nbsp;·&nbsp;
                <span [style.color]="clearResult()!.error ? 'var(--sapNegativeColor,#bb0000)' : 'var(--sapPositiveColor,#188918)'">
                  {{ clearResult()!.error ?? (clearResult()!.deleted + ' HANA rows · ' + clearResult()!.filesDeleted + ' files deleted') }}
                </span>
              }
            </div>
          </div>
        </div>
        <button fd-button fdType="negative" glyph="delete"
          label="Clear HANA Logs"
          [disabled]="clearing()"
          (click)="confirmClear()"
          ariaLabel="Delete all rows from HANA table">
        </button>
      </div>

    </div>

    <!-- ── Clear confirmation dialog ─────────────────────────────────── -->
    <ng-template [fdDialogTemplate] let-dialog let-dialogConfig="dialogConfig" #confirmDialog>
      <fd-dialog [dialogConfig]="dialogConfig" [dialogRef]="dialog" style="max-width:480px;">
        <fd-dialog-header>
          <h1 fd-title id="confirm-title">Clear HANA Logs</h1>
        </fd-dialog-header>
        <fd-dialog-body>
          <p style="margin:0;font-size:.9375rem;color:var(--sapTextColor,#32363a)">
            This will permanently delete <strong>all rows</strong> from
            <code>{{ HANA_SCHEMA }}.IBP_OPTEXPLLOG1</code> and
            <code>{{ HANA_SCHEMA }}.IBP_OPTEXPLLOG2</code>,
            and all files in the local downloads folder.
          </p>
          <p style="margin:.75rem 0 0;font-size:.875rem;color:var(--sapNegativeColor,#bb0000)">
            This action cannot be undone.
          </p>
        </fd-dialog-body>
        <fd-dialog-footer>
          <fd-button-bar fdType="negative" label="Delete All Rows"
            (click)="executeClear(); dialog.close()" ariaLabel="Confirm delete"></fd-button-bar>
          <fd-button-bar label="Cancel"
            (click)="dialog.close()" ariaLabel="Cancel"></fd-button-bar>
        </fd-dialog-footer>
      </fd-dialog>
    </ng-template>
    <ng-template [fdDialogTemplate] let-dialog let-dialogConfig="dialogConfig" #csvDialog>
      <fd-dialog [dialogConfig]="dialogConfig" [dialogRef]="dialog"
        style="width:90vw;max-width:1200px;">
        <fd-dialog-header>
          <h1 fd-title id="csv-dialog-title">{{ selectedCsv()?.filename }}</h1>
          <p style="font-size:.75rem;color:var(--sapNeutralTextColor,#6a6d70);margin:.25rem 0 0">
            {{ selectedCsv()?.totalRows | number }} total rows
            · {{ selectedCsv()?.columns?.length }} columns
            · showing first 500
          </p>
        </fd-dialog-header>

        <fd-dialog-body>
          <div fd-scrollbar style="max-height:65vh;overflow:auto;">
            @if (selectedCsv()) {
              <table fd-table aria-label="CSV data preview" style="font-size:.75rem;white-space:nowrap;">
                <thead fd-table-header>
                  <tr fd-table-row>
                    @for (col of selectedCsv()!.columns; track col) {
                      <th fd-table-cell style="font-size:.75rem">{{ col }}</th>
                    }
                  </tr>
                </thead>
                <tbody fd-table-body>
                  @for (row of selectedCsv()!.rows; track $index) {
                    <tr fd-table-row>
                      @for (col of selectedCsv()!.columns; track col) {
                        <td fd-table-cell style="font-size:.75rem;max-width:200px;overflow:hidden;text-overflow:ellipsis">
                          {{ row[col] }}
                        </td>
                      }
                    </tr>
                  }
                </tbody>
              </table>
            }
          </div>
        </fd-dialog-body>

        <fd-dialog-footer>
          <fd-button-bar fdType="emphasized" label="Close"
            (click)="dialog.close()" ariaLabel="Close"></fd-button-bar>
        </fd-dialog-footer>
      </fd-dialog>
    </ng-template>
  `,
  styles: [`
    :host { display:block; background:var(--sapBackgroundColor,#f2f2f2); min-height:100vh; }

    .ibp-card {
      background: var(--sapTile_Background, #fff);
      border: 1px solid var(--sapGroup_TitleBorderColor, #e5e5e5);
      border-radius: .25rem;
      box-shadow: 0 0 .25rem 0 rgba(0,0,0,.06);
    }

    .ibp-badge {
      display: inline-flex; align-items: center; justify-content: center;
      min-width: 1.25rem; height: 1.25rem; padding: 0 .375rem;
      border-radius: .625rem; font-size: .6875rem; font-weight: 700;
      background: var(--sapButton_Emphasized_Background, #0a6ed1);
      color: #fff; margin-left: .5rem; vertical-align: middle;
    }

    .ibp-section-label {
      font-size: .6875rem; font-weight: 700; letter-spacing: .08em;
      text-transform: uppercase; color: var(--sapNeutralTextColor, #6a6d70);
      margin-bottom: .625rem;
    }

    .ibp-csv-chip {
      display: flex; align-items: center; gap: .75rem;
      padding: .875rem 1.25rem; cursor: pointer;
      transition: box-shadow .15s;
      min-width: 200px;
    }
    .ibp-csv-chip:hover { box-shadow: 0 2px 8px rgba(0,0,0,.12); }

    .ibp-active-btn {
      background: var(--sapButton_Selected_Background, #e8f0fa) !important;
      border-color: var(--sapButton_Selected_BorderColor, #0a6ed1) !important;
    }
  `]
})
export class AppComponent implements OnInit, OnDestroy {
  readonly API        = API;
  readonly LOCAL_API  = 'http://localhost:5001/api';
  readonly DEFAULT_JOB = '';
  readonly HANA_SCHEMA = 'AICOE';
  readonly HANA_TABLE  = 'IBP_OPTIMIZER_LOGS';

  @ViewChild('csvDialog')     private csvDialogTemplate!: TemplateRef<unknown>;
  @ViewChild('confirmDialog') private confirmDialogTemplate!: TemplateRef<unknown>;

  jobNameInput  = signal(this.DEFAULT_JOB);
  files         = signal<LogFile[]>([]);
  jobStatus     = signal<JobStatus | null>(null);
  selectedCsv   = signal<CsvData | null>(null);
  canExtract         = signal(false);
  kymaMode           = signal(false);
  sessionInfo        = signal<SessionInfo | null>(null);
  uploadingCookies   = signal(false);
  cookieUploadResult = signal<{ error?: string; count?: number } | null>(null);
  pushingSession     = signal(false);
  pushResult         = signal<{ error?: string; localSizeMB?: number } | null>(null);
  isLocalBackend     = signal(true);
  clearing           = signal(false);
  clearResult = signal<{ deleted?: number; filesDeleted?: number; error?: string } | null>(null);
  activeCsvKey       = signal<string>('');

  isRunning   = computed(() => this.jobStatus()?.status === 'running');
  lastMessage = computed(() => this.jobStatus()?.message ?? '');
  runLog      = computed(() => this.jobStatus()?.log ?? []);

  hanaEntries = computed(() => {
    const hs = this.jobStatus()?.hanaStatus;
    if (!hs) return [];
    return Object.entries(hs)
      .filter(([k]) => k !== 'error')
      .map(([file, v]: [string, any]) => ({ file, inserted: v.inserted, table: v.table ?? '' }));
  });

  statusLabel = computed(() => {
    const s = this.jobStatus()?.status;
    if (s === 'running') return 'Running';
    if (s === 'done')    return 'Completed';
    if (s === 'error')   return 'Failed';
    return 'Ready';
  });

  statusIcon = computed(() => {
    const s = this.jobStatus()?.status;
    if (s === 'running') return 'synchronize';
    if (s === 'done')    return 'accept';
    if (s === 'error')   return 'message-error';
    return 'status-positive';
  });

  messageType = computed((): 'error' | 'success' | 'information' | 'warning' => {
    const s = this.jobStatus()?.status;
    if (s === 'error') return 'error';
    if (s === 'done')  return 'success';
    return 'information';
  });

  private readonly dialogService = inject(DialogService);
  private readonly http          = inject(HttpClient);
  private pollSub?: Subscription;

  ngOnInit() {
    // Check local agent (localhost:5001) AND CF backend for canExtract
    this.http.get<ModeInfo>(`${LOCAL_API}/mode`).subscribe({
      next: (m) => { if (m.canExtract) this.canExtract.set(true); },
      error: ()  => {},
    });
    this.http.get<ModeInfo>(`${this.API}/mode`).subscribe({
      next: (m) => {
        if (m.canExtract) this.canExtract.set(true);
        if (m.platform === 'kyma') this.kymaMode.set(true);
      },
      error: () => {},
    });
    // Session status from CF backend
    this.http.get<SessionInfo>(`${this.API}/session/status`).subscribe({
      next: (s) => this.sessionInfo.set(s),
    });
    this.loadFiles();
    // Poll job status — prefer local agent, fall back to CF (Kyma)
    const statusUrl = () => this.kymaMode() ? `${this.API}/status` : `${LOCAL_API}/status`;
    this.http.get<JobStatus>(statusUrl()).subscribe({ next: s => this.jobStatus.set(s), error: () => {} });
    this.pollSub = interval(3000).pipe(
      switchMap(() => this.http.get<JobStatus>(statusUrl()))
    ).subscribe({
      next: (s) => {
        const wasRunning = this.jobStatus()?.status === 'running';
        this.jobStatus.set(s);
        if (wasRunning && s.status === 'done') this.loadFiles();
      },
      error: () => {}
    });
  }

  ngOnDestroy() { this.pollSub?.unsubscribe(); }

  loadFiles() {
    this.http.get<LogFile[]>(`${API}/files`).subscribe(f => this.files.set(f));
  }

  triggerRun() {
    const jobName = this.jobNameInput().trim();
    if (!jobName) return;
    // Kyma mode: trigger via CF backend. Local mode: trigger via local agent.
    const runUrl = this.kymaMode() ? `${this.API}/run` : `${LOCAL_API}/run`;
    this.http.post(runUrl, { jobName }).subscribe({
      error: (e) => this.jobStatus.set({
        status: 'error',
        message: e.error?.error ?? 'Failed to start extraction.',
        file: null, csvFiles: [],
        log: this.jobStatus()?.log ?? [],
        hanaStatus: null,
        jobName,
      })
    });
  }

  uploadCookies(event: Event) {
    const input = event.target as HTMLInputElement;
    const file  = input.files?.[0];
    if (!file) return;
    this.uploadingCookies.set(true);
    this.cookieUploadResult.set(null);
    const form = new FormData();
    form.append('file', file);
    this.http.post<{ uploaded: boolean; count: number }>(`${this.API}/session/upload`, form).subscribe({
      next: (r) => {
        this.cookieUploadResult.set({ count: r.count });
        this.uploadingCookies.set(false);
        this.http.get<SessionInfo>(`${this.API}/session/status`).subscribe({
          next: (s) => this.sessionInfo.set(s),
        });
        input.value = '';
      },
      error: (e) => {
        this.cookieUploadResult.set({ error: e.error?.error ?? 'Upload failed.' });
        this.uploadingCookies.set(false);
        input.value = '';
      },
    });
  }

  pushSession() {
    this.pushingSession.set(true);
    this.pushResult.set(null);
    // Always call LOCAL_API — the local Flask backend zips and pushes to CF
    this.http.post<{ pushed: boolean; localSizeMB: number }>(`${this.LOCAL_API}/session/push`, {}).subscribe({
      next: (r) => {
        this.pushResult.set({ localSizeMB: r.localSizeMB });
        this.pushingSession.set(false);
        // Refresh session status from CF backend
        this.http.get<SessionInfo>(`${this.API}/session/status`).subscribe({
          next: (s) => this.sessionInfo.set(s),
        });
      },
      error: (e) => {
        this.pushResult.set({ error: e.error?.error ?? 'Could not reach local backend. Make sure ./start.sh is running.' });
        this.pushingSession.set(false);
      },
    });
  }

  confirmClear() {
    this.clearResult.set(null);
    this.dialogService.open(this.confirmDialogTemplate, {
      ariaLabelledBy: 'confirm-title',
      responsivePadding: true,
    });
  }

  executeClear() {
    this.clearing.set(true);
    this.http.delete<{ deleted: number; filesDeleted: number }>(`${API}/hana/clear`).subscribe({
      next: (r) => {
        this.clearResult.set({ deleted: r.deleted, filesDeleted: r.filesDeleted });
        this.clearing.set(false);
        this.files.set([]);
        this.selectedCsv.set(null);
        this.activeCsvKey.set('');
      },
      error: (e) => {
        this.clearResult.set({ error: e.error?.error ?? 'Failed to clear.' });
        this.clearing.set(false);
      },
    });
  }

  uploadFile(event: Event) {
    const input = event.target as HTMLInputElement;
    const file  = input.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    form.append('jobName', this.jobNameInput());
    this.jobStatus.set({
      status: 'running', message: `Uploading ${file.name}…`,
      file: null, csvFiles: [], log: [], hanaStatus: null, jobName: this.jobNameInput(),
    });
    this.http.post<{saved: string; hanaStatus: Record<string, unknown>; csvFiles: unknown}>
      (`${API}/upload`, form).subscribe({
      next: (r) => {
        this.jobStatus.set({
          status: 'done', message: `Uploaded and processed: ${r.saved}`,
          file: r.saved, csvFiles: (r.csvFiles as any) || [], log: [],
          hanaStatus: r.hanaStatus, jobName: this.jobNameInput(),
        });
        this.loadFiles();
        input.value = '';
      },
      error: (e) => {
        this.jobStatus.set({
          status: 'error', message: e.error?.error ?? 'Upload failed.',
          file: null, csvFiles: [], log: [], hanaStatus: null, jobName: this.jobNameInput(),
        });
        input.value = '';
      },
    });
  }

  openCsvViewer(zipFilename: string, csvName: string) {
    this.loadCsvInline(zipFilename, csvName);
  }

  loadCsvInline(zipFilename: string, csvName: string) {
    const key = zipFilename + csvName;
    if (this.activeCsvKey() === key) {
      this.selectedCsv.set(null);
      this.activeCsvKey.set('');
      return;
    }
    this.http.get<CsvData>(`${API}/files/${zipFilename}/csv/${csvName}`).subscribe({
      next: (data) => {
        this.selectedCsv.set(data);
        this.activeCsvKey.set(key);
      }
    });
  }

  formatBytes(bytes: number): string {
    if (!bytes) return '0 B';
    if (bytes < 1024)    return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
  }

  formatTime(iso: string): string {
    try {
      return new Date(iso).toLocaleTimeString('en-US', {
        hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
      });
    } catch { return iso; }
  }
}
