import { HttpInterceptorFn } from '@angular/common/http';
import { catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';

export const apiInterceptor: HttpInterceptorFn = (req, next) => {
  // Add headers or modify the request here
  const modifiedRequest = req.clone({
    headers: req.headers.set('Content-Type', 'application/json')
  });

  return next(modifiedRequest).pipe(
    catchError((error) => {
      let errorMessage = 'An error occurred';
      
      if (error.error instanceof ErrorEvent) {
        // Client-side error
        errorMessage = `Error: ${error.error.message}`;
      } else {
        // Server-side error
        errorMessage = `Error Code: ${error.status}\nMessage: ${error.message}`;
      }
      
      console.error('API Error:', errorMessage);
      return throwError(() => error);
    })
  );
};
