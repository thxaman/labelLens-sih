import { ROLES } from "../constants/rbac.js";

/**
 * Custom Error for Row-Level Scoping Violations and Missing Hierarchical Attributes
 */
export class SecurityScopingError extends Error {
  constructor(message, details = {}) {
    super(message);
    this.name = "SecurityScopingError";
    this.statusCode = 403;
    this.details = details;
  }
}

/**
 * Generates the Prisma 'where' clause for Inspection queries
 * based on the user's role and hierarchy attributes.
 *
 * Scoping Matrix:
 * - DIRECTOR: Global access (no restrictive where clauses).
 * - INSPECTOR: Queries must append where: { inspectorId: req.user.id }
 * - REVIEWER: District-wide via inspector relation when district assigned, else { reviewerId: req.user.id }
 * - CONTROLLER: Queries must filter by jurisdiction: state (preferred) or district of the inspector
 * - MANUFACTURER: Queries must append where: { inspectorId: req.user.id }.
 * - CONSUMER: Queries must filter by self: where: { consumerId: req.user.id }
 *
 * @param {import("../constants/rbac.js").AuthenticatedUserContext} user 
 * @returns {Record<string, any>} Prisma where condition
 */
export function getInspectionScope(user) {
  if (!user || !user.role) {
    throw new SecurityScopingError("Authentication context missing for inspection scoping");
  }

  const role = user.role.toUpperCase();

  switch (role) {
    case ROLES.DIRECTOR:
      // Global unrestricted access across all jurisdictions and entities
      return {};

    case ROLES.INSPECTOR:
      if (!user.id) {
        throw new SecurityScopingError("Inspector user ID missing in security context");
      }
      return { inspectorId: user.id };

    case ROLES.REVIEWER:
      // District-wide QA queue when a district is assigned (the reviewer
      // validates all district inspections); otherwise fall back to only
      // inspections formally assigned for their review.
      if (user.district && String(user.district).trim()) {
        return {
          inspector: {
            district: {
              equals: String(user.district).trim(),
              mode: "insensitive",
            },
          },
        };
      }
      if (!user.id) {
        throw new SecurityScopingError("Reviewer user ID missing in security context");
      }
      return { reviewerId: user.id };

    case ROLES.CONTROLLER:
      if (!user.state && !user.district) {
        throw new SecurityScopingError(
          "CONTROLLER account is missing assigned jurisdiction (state or district). Access denied to prevent data leakage.",
          { role, userId: user.id }
        );
      }
      return user.state
        ? { inspector: { state: { equals: String(user.state).trim(), mode: "insensitive" } } }
        : { inspector: { district: { equals: String(user.district).trim(), mode: "insensitive" } } };

    case ROLES.MANUFACTURER:
      if (!user.id) {
        throw new SecurityScopingError("Manufacturer user ID missing in security context");
      }
      return { inspectorId: user.id };

    case ROLES.CONSUMER:
      if (!user.id) {
        throw new SecurityScopingError("Consumer user ID missing in security context");
      }
      return { consumerId: user.id };

    default:
      throw new SecurityScopingError(`Unrecognized role '${role}' cannot be scoped for inspections`, {
        role,
      });
  }
}

/**
 * Generates the Prisma 'where' clause for Complaint queries
 * @param {import("../constants/rbac.js").AuthenticatedUserContext} user 
 * @returns {Record<string, any>} Prisma where condition
 */
export function getComplaintScope(user) {
  if (!user || !user.role) {
    throw new SecurityScopingError("Authentication context missing for complaint scoping");
  }

  const role = user.role.toUpperCase();

  switch (role) {
    case ROLES.DIRECTOR:
      return {};

    case ROLES.CONTROLLER:
      if (!user.state && !user.district) {
        throw new SecurityScopingError("CONTROLLER missing state or district for complaint triage");
      }
      return user.state
        ? { state: { equals: String(user.state).trim(), mode: "insensitive" } }
        : { district: { equals: String(user.district).trim(), mode: "insensitive" } };

    case ROLES.REVIEWER:
      if (!user.district) {
        return { consumerId: user.id };
      }
      // Show district-matched complaints OR unrouted (district=null) complaints
      // so that legacy/unrouted complaints are visible for triage.
      return {
        OR: [
          { district: { equals: String(user.district).trim(), mode: "insensitive" } },
          { district: null },
        ],
      };

    case ROLES.INSPECTOR:
      if (!user.district) {
        return { consumerId: user.id };
      }
      return {
        district: {
          equals: String(user.district).trim(),
          mode: "insensitive",
        },
      };

    case ROLES.CONSUMER:
      return { consumerId: user.id };

    case ROLES.MANUFACTURER:
      // Manufacturers cannot browse arbitrary consumer complaints
      return { id: "__UNAUTHORIZED_COMPLAINT_ACCESS__" };

    default:
      throw new SecurityScopingError(`Role '${role}' cannot access complaints`);
  }
}

/**
 * Generates the Prisma 'where' clause for Report queries
 * @param {import("../constants/rbac.js").AuthenticatedUserContext} user 
 * @returns {Record<string, any>}
 */
export function getReportScope(user) {
  if (!user || !user.role) {
    throw new SecurityScopingError("Authentication context missing for report scoping");
  }

  const role = user.role.toUpperCase();

  switch (role) {
    case ROLES.DIRECTOR:
      return {};

    case ROLES.CONTROLLER:
      if (!user.state && !user.district) {
        throw new SecurityScopingError("CONTROLLER missing jurisdiction state or district for report scoping");
      }
      return user.state
        ? { inspection: { inspector: { state: { equals: String(user.state).trim(), mode: "insensitive" } } } }
        : { inspection: { inspector: { district: { equals: String(user.district).trim(), mode: "insensitive" } } } };

    case ROLES.INSPECTOR:
      return {
        OR: [
          { generatedById: user.id },
          { inspection: { inspectorId: user.id } },
        ],
      };

    case ROLES.REVIEWER:
      return {
        OR: [
          { generatedById: user.id },
          { inspection: { reviewerId: user.id } },
        ],
      };

    case ROLES.MANUFACTURER:
      return {
        OR: [
          { generatedById: user.id },
          { inspection: { inspectorId: user.id } },
        ],
      };

    case ROLES.CONSUMER:
      return {
        inspection: {
          consumerId: user.id,
        },
      };

    default:
      throw new SecurityScopingError(`Role '${role}' cannot access reports`);
  }
}

/**
 * Securely merges a mandatory row-level scope with any additional user-provided query conditions.
 * Uses Prisma's AND array to ensure client filters can never override or widen the security scope.
 *
 * @param {Record<string, any>} securityScope Mandatory scope generated by scoping service
 * @param {Record<string, any>} [userWhere] Additional filters requested by user
 * @returns {Record<string, any>} Combined Prisma where object
 */
export function mergeScope(securityScope, userWhere = {}) {
  const hasSecurityScope = securityScope && Object.keys(securityScope).length > 0;
  const hasUserWhere = userWhere && Object.keys(userWhere).length > 0;

  if (!hasSecurityScope && !hasUserWhere) {
    return {};
  }
  if (!hasSecurityScope) {
    return { ...userWhere };
  }
  if (!hasUserWhere) {
    return { ...securityScope };
  }

  return {
    AND: [securityScope, userWhere],
  };
}

/**
 * In-memory entity access validator.
 * Validates whether an already-fetched entity record is within the caller's permitted scope.
 * Prevents Insecure Direct Object Reference (IDOR) attacks.
 *
 * @param {import("../constants/rbac.js").AuthenticatedUserContext} user
 * @param {Record<string, any>} inspection Inspection record with relations populated
 * @returns {boolean}
 */
export function canAccessInspection(user, inspection) {
  if (!user || !inspection) return false;
  const role = (user.role || "").toUpperCase();

  switch (role) {
    case ROLES.DIRECTOR:
      return true;

    case ROLES.INSPECTOR:
      return inspection.inspectorId === user.id;

    case ROLES.REVIEWER:
      if (inspection.reviewerId === user.id) return true;
      return Boolean(
        user.district &&
        inspection.inspector?.district &&
        inspection.inspector.district.toLowerCase().trim() === user.district.toLowerCase().trim()
      );

    case ROLES.CONTROLLER: {
      if (user.state && inspection.inspector?.state) {
        return inspection.inspector.state.toLowerCase().trim() === user.state.toLowerCase().trim();
      }
      if (!user.district) return false;
      const inspectorDistrict = inspection.inspector?.district;
      return (
        typeof inspectorDistrict === "string" &&
        inspectorDistrict.toLowerCase().trim() === user.district.toLowerCase().trim()
      );
    }

    case ROLES.MANUFACTURER:
      return inspection.inspectorId === user.id;

    case ROLES.CONSUMER:
      return inspection.consumerId === user.id;

    default:
      return false;
  }
}

/**
 * Asserts access to an inspection entity; throws SecurityScopingError if unauthorized.
 *
 * @param {import("../constants/rbac.js").AuthenticatedUserContext} user 
 * @param {Record<string, any>} inspection 
 */
export function assertInspectionAccess(user, inspection) {
  if (!canAccessInspection(user, inspection)) {
    throw new SecurityScopingError(
      `Access denied: Record does not belong to your jurisdictional or organizational scope.`,
      {
        userId: user?.id,
        role: user?.role,
        inspectionId: inspection?.id,
      }
    );
  }
}
