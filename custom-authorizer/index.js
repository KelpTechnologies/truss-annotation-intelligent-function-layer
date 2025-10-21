// custom-authorizer/index.js
const jwt = require("jsonwebtoken");
const jwksClient = require("jwks-rsa");

// Configuration for supported user pools
const SUPPORTED_USER_POOLS = [
  {
    userPoolId: "eu-west-2_JyaPgaRFW",
    region: "eu-west-2",
    arn: "arn:aws:cognito-idp:eu-west-2:193757560043:userpool/eu-west-2_JyaPgaRFW",
  },
  {
    userPoolId: "eu-west-2_kTdl3FlEo",
    region: "eu-west-2",
    arn: "arn:aws:cognito-idp:eu-west-2:193757560043:userpool/eu-west-2_kTdl3FlEo",
  },
  {
    userPoolId: "eu-west-2_uwZ2LBjRY",
    region: "eu-west-2",
    arn: "arn:aws:cognito-idp:eu-west-2:193757560043:userpool/eu-west-2_uwZ2LBjRY",
  },
];

// Cache for JWKS clients
const jwksClients = {};

// Initialize JWKS clients for each user pool
SUPPORTED_USER_POOLS.forEach((pool) => {
  const jwksUri = `https://cognito-idp.${pool.region}.amazonaws.com/${pool.userPoolId}/.well-known/jwks.json`;
  jwksClients[pool.userPoolId] = jwksClient({
    jwksUri,
    cache: true,
    cacheMaxAge: 86400000, // 24 hours
    rateLimit: true,
    jwksRequestsPerMinute: 10,
  });
});

/**
 * Get signing key for JWT verification
 */
function getKey(header, userPoolId, callback) {
  const client = jwksClients[userPoolId];
  if (!client) {
    return callback(
      new Error(`No JWKS client found for user pool: ${userPoolId}`)
    );
  }

  client.getSigningKey(header.kid, (err, key) => {
    if (err) {
      return callback(err);
    }
    const signingKey = key.publicKey || key.rsaPublicKey;
    callback(null, signingKey);
  });
}

/**
 * Verify JWT token against a specific user pool
 */
function verifyToken(token, userPool) {
  return new Promise((resolve, reject) => {
    // Decode token header to get key ID
    const decodedHeader = jwt.decode(token, { complete: true });
    if (!decodedHeader) {
      return reject(new Error("Invalid token format"));
    }

    // Get signing key
    getKey(decodedHeader.header, userPool.userPoolId, (err, signingKey) => {
      if (err) {
        return reject(err);
      }

      // Verify token
      const verifyOptions = {
        issuer: `https://cognito-idp.${userPool.region}.amazonaws.com/${userPool.userPoolId}`,
        algorithms: ["RS256"],
      };

      jwt.verify(token, signingKey, verifyOptions, (err, decoded) => {
        if (err) {
          return reject(err);
        }

        // Additional validation
        if (decoded.token_use !== "access" && decoded.token_use !== "id") {
          return reject(new Error("Invalid token use"));
        }

        resolve({
          decoded,
          userPool: userPool,
        });
      });
    });
  });
}

/**
 * Generate IAM policy for API Gateway
 */
function generatePolicy(principalId, effect, resource, context = {}) {
  const authResponse = {
    principalId: principalId,
    policyDocument: {
      Version: "2012-10-17",
      Statement: [
        {
          Action: "execute-api:Invoke",
          Effect: effect,
          Resource: resource,
        },
      ],
    },
    context: context,
  };

  return authResponse;
}

/**
 * Main authorizer handler
 */
exports.handler = async (event) => {
  console.log("Custom authorizer invoked:", JSON.stringify(event, null, 2));

  try {
    // Extract token from Authorization header
    const token = event.authorizationToken;
    if (!token) {
      throw new Error("Missing authorization token");
    }

    // Remove 'Bearer ' prefix if present
    const cleanToken = token.replace(/^Bearer\s+/i, "");

    // Try to verify token against each supported user pool
    let verificationResult = null;
    let lastError = null;

    for (const userPool of SUPPORTED_USER_POOLS) {
      try {
        console.log(
          `Attempting verification against user pool: ${userPool.userPoolId}`
        );
        verificationResult = await verifyToken(cleanToken, userPool);
        console.log(
          `Token verified successfully against user pool: ${userPool.userPoolId}`
        );
        break;
      } catch (error) {
        console.log(
          `Verification failed for user pool ${userPool.userPoolId}:`,
          error.message
        );
        lastError = error;
        continue;
      }
    }

    // If no user pool accepted the token, deny access
    if (!verificationResult) {
      console.error(
        "Token verification failed against all user pools:",
        lastError?.message
      );
      throw new Error("Unauthorized");
    }

    const { decoded, userPool } = verificationResult;

    // Extract user information
    const principalId = decoded.sub || decoded.username || "user";
    const username =
      decoded.username || decoded["cognito:username"] || principalId;
    const email = decoded.email;
    const groups = decoded["cognito:groups"] || [];

    // Create context to pass to the Lambda function
    const context = {
      userId: principalId,
      username: username,
      email: email || "",
      groups: JSON.stringify(groups),
      userPoolId: userPool.userPoolId,
      userPoolArn: userPool.arn,
      tokenUse: decoded.token_use,
      authTime: decoded.auth_time?.toString() || "",
      iat: decoded.iat?.toString() || "",
      exp: decoded.exp?.toString() || "",
    };

    console.log("Authorization successful for user:", {
      principalId,
      username,
      userPoolId: userPool.userPoolId,
      groups,
    });

    // Generate allow policy with wildcard to prevent route locking
    const wildcardResource = event.methodArn.replace(/\/[^/]+\/.*$/, "/*/*");
    const policy = generatePolicy(
      principalId,
      "Allow",
      wildcardResource,
      context
    );
    return policy;
  } catch (error) {
    console.error("Authorization error:", error.message);

    // For debugging purposes, you might want to return more specific error information
    // In production, you typically just return 'Unauthorized'
    throw new Error("Unauthorized");
  }
};

/**
 * Health check function for monitoring
 */
exports.healthCheck = async () => {
  const health = {
    status: "healthy",
    timestamp: new Date().toISOString(),
    supportedUserPools: SUPPORTED_USER_POOLS.map((pool) => ({
      userPoolId: pool.userPoolId,
      region: pool.region,
    })),
    jwksClientsInitialized: Object.keys(jwksClients).length,
  };

  return health;
};
