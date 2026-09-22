import prisma from "../config/db.js";
import { getComplaintScope, mergeScope, SecurityScopingError } from "./dataScopingService.js";
import { ROLES } from "../constants/rbac.js";

/**
 * File a complaint (Consumers)
 *
 * @param {import("../constants/rbac.js").AuthenticatedUserContext} user 
 * @param {Object} data 
 */
export async function fileComplaint(user, data) {
  if (!user || user.role !== ROLES.CONSUMER) {
    throw new SecurityScopingError("Only verified consumers can file complaints");
  }

  const { title, description, inspectionId, district, state } = data;

  if (!title || !description) {
    throw new Error("Title and description are required for filing a complaint");
  }

  // Resolve district/state: explicit > consumer profile > inspector's district (from linked inspection)
  let resolvedDistrict = district || user.district || null;
  let resolvedState = state || user.state || null;

  if ((!resolvedDistrict || !resolvedState) && inspectionId) {
    const linkedInspection = await prisma.inspection.findUnique({
      where: { id: inspectionId },
      select: { inspector: { select: { district: true, state: true } } },
    });
    if (linkedInspection?.inspector) {
      resolvedDistrict = resolvedDistrict || linkedInspection.inspector.district || null;
      resolvedState = resolvedState || linkedInspection.inspector.state || null;
    }
  }

  const complaint = await prisma.complaint.create({
    data: {
      consumerId: user.id,
      inspectionId: inspectionId || null,
      title: String(title).trim(),
      description: String(description).trim(),
      district: resolvedDistrict,
      state: resolvedState,
      status: "PENDING",
    },
    include: {
      consumer: {
        select: {
          id: true,
          fullName: true,
          email: true,
        },
      },
      inspection: {
        select: {
          id: true,
          status: true,
          imagePath: true,
        },
      },
    },
  });

  return complaint;
}

/**
 * Lists complaints scoped to the user's role and district
 *
 * @param {import("../constants/rbac.js").AuthenticatedUserContext} user 
 * @param {Object} [queryOptions] 
 */
export async function listComplaints(user, queryOptions = {}) {
  const page = Math.max(1, parseInt(queryOptions.page) || 1);
  const limit = Math.min(100, Math.max(1, parseInt(queryOptions.limit) || 20));
  const skip = (page - 1) * limit;

  const securityScope = getComplaintScope(user);

  const userWhere = {};
  if (queryOptions.status) {
    userWhere.status = String(queryOptions.status).toUpperCase();
  }

  const where = mergeScope(securityScope, userWhere);

  const [total, items] = await Promise.all([
    prisma.complaint.count({ where }),
    prisma.complaint.findMany({
      where,
      skip,
      take: limit,
      orderBy: { createdAt: "desc" },
      include: {
        consumer: {
          select: {
            id: true,
            fullName: true,
            email: true,
          },
        },
        inspection: {
          select: {
            id: true,
            complianceScore: true,
            status: true,
          },
        },
      },
    }),
  ]);

  return {
    page,
    limit,
    total,
    total_pages: Math.ceil(total / limit),
    items,
  };
}

/**
 * Triage or update a complaint status (Controller / Consumer triage)
 *
 * @param {import("../constants/rbac.js").AuthenticatedUserContext} user 
 * @param {string} complaintId 
 * @param {Object} triageData 
 */
export async function triageComplaint(user, complaintId, triageData = {}) {
  const securityScope = getComplaintScope(user);
  const complaint = await prisma.complaint.findFirst({
    where: mergeScope(securityScope, { id: complaintId }),
  });

  if (!complaint) {
    throw new SecurityScopingError(
      `Complaint '${complaintId}' not found or outside your jurisdictional scope`
    );
  }

  const updated = await prisma.complaint.update({
    where: { id: complaintId },
    data: {
      status: triageData.status ? String(triageData.status).toUpperCase() : complaint.status,
    },
  });

  return updated;
}
