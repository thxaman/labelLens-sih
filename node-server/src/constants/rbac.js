/**
 * Role-Based Access Control (RBAC) Constants & Permission Matrix
 * Legal Metrology (LMPC) Compliance System
 */

export const ROLES = Object.freeze({
  DIRECTOR: "DIRECTOR",
  CONTROLLER: "CONTROLLER",
  REVIEWER: "REVIEWER",
  INSPECTOR: "INSPECTOR",
  MANUFACTURER: "MANUFACTURER",
  CONSUMER: "CONSUMER",
});

export const ALL_ROLES = Object.freeze(Object.values(ROLES));

export const PERMISSIONS = Object.freeze({
  USER_MANAGE: "user:manage",
  ROLE_MANAGE: "role:manage",
  RULE_MANAGE: "rule:manage",
  AUDIT_VIEW: "audit:view",
  DASHBOARD_GLOBAL: "dashboard:global",
  DASHBOARD_JURISDICTION: "dashboard:jurisdiction",
  REPORT_APPROVE: "report:approve",
  REPORT_EXPORT: "report:export",
  REPORT_DRAFT: "report:draft",
  SCAN_VIEW_JURISDICTION: "scan:view_jurisdiction",
  SCAN_VIEW_ORG: "scan:view_org",
  SCAN_VIEW_OWN: "scan:view_own",
  SCAN_CREATE: "scan:create",
  EXTRACTION_EDIT: "extraction:edit",
  VIOLATION_CONFIRM: "violation:confirm",
  PRODUCT_SEARCH: "product:search",
  COMPLAINT_FILE: "complaint:file",
  COMPLAINT_TRIAGE: "complaint:triage",
});

export const ALL_PERMISSIONS = Object.freeze(Object.values(PERMISSIONS));

/**
 * Explicit Role-to-Permissions Mapping
 */
export const ROLE_PERMISSIONS = Object.freeze({
  [ROLES.DIRECTOR]: Object.freeze([
    PERMISSIONS.USER_MANAGE,
    PERMISSIONS.ROLE_MANAGE,
    PERMISSIONS.RULE_MANAGE,
    PERMISSIONS.AUDIT_VIEW,
    PERMISSIONS.DASHBOARD_GLOBAL,
    PERMISSIONS.SCAN_CREATE,
  ]),

  [ROLES.CONTROLLER]: Object.freeze([
    PERMISSIONS.DASHBOARD_JURISDICTION,
    PERMISSIONS.REPORT_APPROVE,
    PERMISSIONS.REPORT_EXPORT,
    PERMISSIONS.SCAN_VIEW_JURISDICTION,
    PERMISSIONS.SCAN_CREATE,
    PERMISSIONS.COMPLAINT_TRIAGE,
  ]),

  [ROLES.REVIEWER]: Object.freeze([
    PERMISSIONS.SCAN_VIEW_ORG,
    PERMISSIONS.VIOLATION_CONFIRM,
    PERMISSIONS.REPORT_DRAFT,
    PERMISSIONS.SCAN_CREATE,
    PERMISSIONS.COMPLAINT_TRIAGE,
  ]),

  [ROLES.INSPECTOR]: Object.freeze([
    PERMISSIONS.SCAN_CREATE,
    PERMISSIONS.SCAN_VIEW_OWN,
    PERMISSIONS.EXTRACTION_EDIT,
    PERMISSIONS.REPORT_DRAFT,
  ]),

  [ROLES.MANUFACTURER]: Object.freeze([
    PERMISSIONS.SCAN_VIEW_OWN,
    PERMISSIONS.SCAN_CREATE,
    PERMISSIONS.PRODUCT_SEARCH,
  ]),

  [ROLES.CONSUMER]: Object.freeze([
    PERMISSIONS.COMPLAINT_FILE,
    PERMISSIONS.COMPLAINT_TRIAGE,
    PERMISSIONS.PRODUCT_SEARCH,
    PERMISSIONS.SCAN_CREATE,
  ]),
});

/**
 * Quick-lookup Set map for O(1) permission validation
 */
const ROLE_PERMISSION_SETS = Object.freeze(
  Object.fromEntries(
    Object.entries(ROLE_PERMISSIONS).map(([role, perms]) => [role, new Set(perms)])
  )
);

/**
 * Roles that belong to the government Legal Metrology Department hierarchy
 */
export const GOVERNMENT_ROLES = Object.freeze([
  ROLES.DIRECTOR,
  ROLES.CONTROLLER,
  ROLES.REVIEWER,
  ROLES.INSPECTOR,
]);

/**
 * Check if a role string is a valid UserRole
 * @param {string} role 
 * @returns {boolean}
 */
export function isValidRole(role) {
  return typeof role === "string" && ALL_ROLES.includes(role.toUpperCase());
}

/**
 * Check if a role belongs to government personnel
 * @param {string} role 
 * @returns {boolean}
 */
export function isGovernmentRole(role) {
  return typeof role === "string" && GOVERNMENT_ROLES.includes(role.toUpperCase());
}

/**
 * Retrieve the array of permissions granted to a given role
 * @param {string} role 
 * @returns {readonly string[]}
 */
export function getRolePermissions(role) {
  if (!role) return [];
  const normalized = role.toUpperCase();
  return ROLE_PERMISSIONS[normalized] || [];
}

/**
 * Check if a role has a specific permission
 * @param {string} role 
 * @param {string} permission 
 * @returns {boolean}
 */
export function hasPermission(role, permission) {
  if (!role || !permission) return false;
  const set = ROLE_PERMISSION_SETS[role.toUpperCase()];
  return !!set && set.has(permission);
}

/**
 * Check if a role has ALL of the requested permissions
 * @param {string} role 
 * @param {string[]} permissions 
 * @returns {boolean}
 */
export function hasAllPermissions(role, permissions = []) {
  if (!role || !Array.isArray(permissions)) return false;
  const set = ROLE_PERMISSION_SETS[role.toUpperCase()];
  if (!set) return false;
  return permissions.every((p) => set.has(p));
}

/**
 * Check if a role has AT LEAST ONE of the requested permissions
 * @param {string} role 
 * @param {string[]} permissions 
 * @returns {boolean}
 */
export function hasAnyPermission(role, permissions = []) {
  if (!role || !Array.isArray(permissions)) return false;
  const set = ROLE_PERMISSION_SETS[role.toUpperCase()];
  if (!set) return false;
  return permissions.some((p) => set.has(p));
}
